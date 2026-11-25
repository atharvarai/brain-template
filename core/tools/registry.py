"""All tools from all sources behind one object: discovery, routing, limits, failures.

The pipeline (and later the agent) sees only ``ToolExecutor``: the tools on offer and a way
to run a call. Whatever goes wrong while running a tool becomes an error ``ToolResult`` for
the model to read, never an exception that ends the user's request.
"""

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self

import anyio

from core.llm import ToolCall, ToolResult, ToolSpec
from core.tools.errors import ToolError
from core.tools.source import ToolInfo, ToolOutput, ToolSource

logger = logging.getLogger(__name__)


class ToolExecutor(Protocol):
    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """The tools the model may call."""
        ...

    def is_read_only(self, name: str) -> bool:
        """True if the tool's server declares that it changes nothing."""
        ...

    async def run(self, call: ToolCall) -> ToolResult: ...


class ToolRegistry:
    def __init__(
        self,
        sources: Sequence[ToolSource] = (),
        timeout_seconds: float = 15.0,
        max_result_chars: int = 4000,
    ) -> None:
        self._sources = list(sources)
        self._timeout = timeout_seconds
        self._max_chars = max_result_chars
        self._routes: dict[str, ToolSource] = {}
        self._specs: list[ToolSpec] = []
        self._read_only: set[str] = set()
        self._opened: list[ToolSource] = []
        self.unavailable: list[str] = []  # sources that could not be reached at start-up

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs)

    def is_read_only(self, name: str) -> bool:
        return name in self._read_only

    async def open(self) -> None:
        """Connect every source and discover its tools.

        A source that cannot be reached is logged and skipped, so the brain still answers
        from its documents. Two sources offering the same tool name is a set-up error.
        """
        self.unavailable = []
        try:
            for source in self._sources:
                try:
                    specs = await source.open()
                except Exception as exc:
                    logger.warning(
                        "tool server '%s' is unavailable, its tools are off: %r", source.name, exc
                    )
                    self.unavailable.append(source.name)
                    continue
                self._opened.append(source)
                self._add(source, specs)
                logger.info("tool server '%s': %d tools", source.name, len(specs))
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        opened, self._opened = self._opened, []
        for source in reversed(opened):
            try:
                await source.close()
            except Exception:
                logger.exception("could not close tool server '%s'", source.name)
        self._routes.clear()
        self._specs.clear()
        self._read_only.clear()

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def run(self, call: ToolCall) -> ToolResult:
        source = self._routes.get(call.name)
        if source is None:
            known = ", ".join(sorted(self._routes)) or "none"
            return self._error(call, f"Unknown tool '{call.name}'. Available tools: {known}.")
        try:
            with anyio.fail_after(self._timeout):
                output = await source.call(call.name, call.arguments)
        except TimeoutError:
            logger.warning("tool '%s' timed out after %ss", call.name, self._timeout)
            return self._error(call, f"The tool did not answer within {self._timeout:g} seconds.")
        except Exception:
            # The details stay in the log: they can hold internal addresses or stack traces.
            logger.exception("tool '%s' on '%s' failed", call.name, source.name)
            return self._error(call, "The tool could not be reached. Try again later.")
        return ToolResult(call.id, call.name, self._limit(output), output.is_error)

    def _add(self, source: ToolSource, tools: Sequence[ToolInfo]) -> None:
        for tool in tools:
            name = tool.spec.name
            owner = self._routes.get(name)
            if owner is not None:
                raise ToolError(
                    f"tool '{name}' is offered by both '{owner.name}' and '{source.name}'"
                )
            self._routes[name] = source
            self._specs.append(tool.spec)
            if tool.read_only:
                self._read_only.add(name)

    def _limit(self, output: ToolOutput) -> str:
        """Cut long results: they cost tokens on every later turn and bury the useful part."""
        text = output.content
        if len(text) <= self._max_chars:
            return text
        cut = len(text) - self._max_chars
        return f"{text[: self._max_chars]}\n[result cut: {cut} more characters not shown]"

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(call.id, call.name, message, is_error=True)
