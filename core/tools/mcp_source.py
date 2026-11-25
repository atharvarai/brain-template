"""MCP adapter for ``ToolSource``, on the official MCP Python SDK.

One long-lived client session per server. The SDK's client must be entered and exited in
the same task, so the session lives in a supervisor task of its own: requests use the open
session, and when a call finds the connection dead the supervisor reconnects in the
background with growing pauses. A server restarted while the brain runs is picked up
again without restarting the brain.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any

import anyio
import httpx2
from anyio.abc import TaskStatus
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, CallToolResult, Implementation, RequestParamsMeta

from core import __version__
from core.llm import ToolSpec
from core.tools.source import ToolInfo, ToolOutput

logger = logging.getLogger(__name__)

# Over stdio there are no headers, so the user's token travels in the request's ``_meta``
# under this key (local development only). Over HTTP it is the Authorization header.
AUTH_META_KEY = "io.brain/authorization"

# A server address: a Streamable HTTP URL, a command to start over stdio, or (in tests) a
# server object connected in-process.
McpTarget = str | StdioServerParameters | MCPServer
_MAX_PAGES = 50  # stop paging a tool listing that never ends


class McpToolSource:
    def __init__(
        self,
        name: str,
        target: McpTarget,
        timeout_seconds: float = 15.0,
        reconnect_wait_seconds: float = 2.0,  # how long a call waits for a reconnect
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self._name = name
        self._target = target
        self._timeout = timeout_seconds
        self._reconnect_wait = reconnect_wait_seconds
        self._max_backoff = max_backoff_seconds
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._ready = anyio.Event()  # set while a session is open
        self._broken = anyio.Event()  # set by a call that found the session dead

    @property
    def name(self) -> str:
        return self._name

    async def open(self) -> Sequence[ToolInfo]:
        """Connect and discover. Raises if the first connection fails."""
        stack = AsyncExitStack()
        try:
            tasks = await stack.enter_async_context(anyio.create_task_group())
            specs: Sequence[ToolInfo] = await tasks.start(self._supervise)
            stack.callback(tasks.cancel_scope.cancel)  # on close, stop the supervisor
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return specs

    async def call(
        self, tool: str, arguments: Mapping[str, Any], credentials: str | None = None
    ) -> ToolOutput:
        meta = _auth_meta(credentials)
        if credentials and isinstance(self._target, str):
            return await self._call_as_user(tool, arguments, credentials, meta)
        client = await self._session()
        try:
            result = await client.call_tool(
                tool, dict(arguments), read_timeout_seconds=self._timeout, meta=meta
            )
        except Exception as exc:
            if _is_disconnect(exc):
                self._mark_down()
                self._broken.set()  # the supervisor reconnects; this call still fails
            raise
        return ToolOutput(_text(result), bool(result.is_error))

    async def _call_as_user(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        credentials: str,
        meta: RequestParamsMeta | None,
    ) -> ToolOutput:
        """Over HTTP, a call made for a user gets its own connection carrying the user's
        token in the Authorization header, the way the server expects any client to.
        The shared session (no user) is kept for discovery."""
        assert isinstance(self._target, str)
        headers = {"Authorization": f"Bearer {credentials}"}
        async with (
            httpx2.AsyncClient(headers=headers, timeout=self._timeout) as http,
            Client(
                streamable_http_client(self._target, http_client=http),
                read_timeout_seconds=self._timeout,
                client_info=Implementation(name="brain-template", version=__version__),
            ) as client,
        ):
            result = await client.call_tool(
                tool, dict(arguments), read_timeout_seconds=self._timeout, meta=meta
            )
        return ToolOutput(_text(result), bool(result.is_error))

    async def close(self) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    async def _session(self) -> Client:
        if self._stack is None:
            raise RuntimeError(f"MCP server '{self._name}' is not connected")
        with anyio.move_on_after(self._reconnect_wait):
            await self._ready.wait()
        if self._client is None:
            raise ConnectionError(f"MCP server '{self._name}' is reconnecting")
        return self._client

    async def _supervise(self, *, task_status: TaskStatus[Sequence[ToolInfo]]) -> None:
        """Own the session: open it, wait until a call reports it dead, reopen it."""
        first, backoff = True, 1.0
        while True:
            try:
                async with self._new_client() as client:
                    specs = await self._discover(client)
                    self._client, backoff = client, 1.0
                    self._ready.set()
                    if first:
                        first = False
                        task_status.started(specs)
                    else:
                        logger.info("tool server '%s' reconnected", self._name)
                    await self._broken.wait()
                    logger.warning("tool server '%s' connection lost, reconnecting", self._name)
            except Exception as exc:
                if first:
                    raise  # start-up failure: the registry marks the server unavailable
                logger.warning("tool server '%s' still unreachable: %r", self._name, exc)
            finally:
                self._mark_down()
                self._broken = anyio.Event()
            await anyio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    def _mark_down(self) -> None:
        """Stop handing out the session at once. The ready event is replaced only if set, so
        calls already waiting for the reconnect are woken by it."""
        self._client = None
        if self._ready.is_set():
            self._ready = anyio.Event()

    def _new_client(self) -> Client:
        return Client(
            self._target,
            read_timeout_seconds=self._timeout,
            client_info=Implementation(name="brain-template", version=__version__),
        )

    @staticmethod
    async def _discover(client: Client) -> list[ToolInfo]:
        specs: list[ToolInfo] = []
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            page = await client.list_tools(cursor=cursor)
            specs.extend(
                ToolInfo(
                    ToolSpec(tool.name, tool.description or "", dict(tool.input_schema)),
                    read_only=bool(tool.annotations and tool.annotations.read_only_hint),
                )
                for tool in page.tools
            )
            cursor = page.next_cursor
            if cursor is None:
                break
        return specs


def _auth_meta(credentials: str | None) -> RequestParamsMeta | None:
    if not credentials:
        return None
    meta: RequestParamsMeta = {}
    meta[AUTH_META_KEY] = f"Bearer {credentials}"  # type: ignore[literal-required]
    return meta


def _is_disconnect(exc: Exception) -> bool:
    """A dead connection, as opposed to the server rejecting one request."""
    if isinstance(exc, MCPError):
        return exc.error.code == CONNECTION_CLOSED
    return not isinstance(exc, TimeoutError)  # transport and OS errors


def _text(result: CallToolResult) -> str:
    """Flatten a tool result to text for the model. Non-text parts are named, not inlined."""
    parts = [
        block.text if block.type == "text" else f"[{block.type} content omitted]"
        for block in result.content
    ]
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    return "\n".join(parts)
