"""Adapter for the local Claude Code CLI (``claude -p``).

Lets someone with a Claude Code subscription run the brain with no API key. It is meant
for development: the CLI takes one prompt per call, so earlier turns are folded into that
prompt, and ``max_tokens`` cannot be enforced.

Tool calling is emulated. The CLI runs tools itself, inside its own loop, which would take
access checks and approvals away from the brain. So the tools are described in the system
prompt instead, and the model asks for them by writing a ``<function_calls>`` block in its
reply. The adapter parses the block into tool calls and keeps it out of the streamed text;
the brain then runs the calls, exactly as it does for vendors with native tool calling.

An earlier version constrained the reply with ``--json-schema`` instead. The evals showed
that it thinned answers (the whole answer had to fit in a JSON string), doubled input
tokens and ruled out streaming, so the plain-text protocol replaced it.
"""

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from core.config import ModelSpec
from core.llm.errors import ProviderRequestError, ProviderUnavailableError, error_for_status
from core.llm.types import (
    Completion,
    LLMRequest,
    Message,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolSpec,
    Usage,
)
from core.prompts.template import wrap_untrusted

_STOP_REASONS = {
    "end_turn": StopReason.END,
    "stop_sequence": StopReason.END,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}
_STDOUT_LIMIT = 16 * 1024 * 1024  # the CLI can emit very long single lines


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Runs the CLI. A port of its own so tests never spawn a real process."""

    async def run(self, args: Sequence[str], stdin: str, timeout: float) -> CommandResult: ...

    def stream_lines(self, args: Sequence[str], stdin: str, timeout: float) -> AsyncIterator[str]:
        """Yield stdout line by line. Raises ``TimeoutError`` if the whole run is too slow."""
        ...


class SubprocessRunner:
    """Real runner. Removes ANTHROPIC_API_KEY so the CLI uses the subscription login."""

    async def run(self, args: Sequence[str], stdin: str, timeout: float) -> CommandResult:
        process = await self._spawn(args)
        try:
            out, err = await asyncio.wait_for(process.communicate(stdin.encode()), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        return CommandResult(process.returncode or 0, out.decode(), err.decode())

    async def stream_lines(
        self, args: Sequence[str], stdin: str, timeout: float
    ) -> AsyncIterator[str]:
        process = await self._spawn(args)
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(stdin.encode())
        await process.stdin.drain()
        process.stdin.close()
        try:
            async with asyncio.timeout(timeout):
                async for raw in process.stdout:
                    yield raw.decode()
                await process.wait()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    @staticmethod
    async def _spawn(args: Sequence[str]) -> asyncio.subprocess.Process:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        return await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=tempfile.gettempdir(),  # a neutral folder: no project settings picked up
            limit=_STDOUT_LIMIT,
        )


OPEN, CLOSE = "<function_calls>", "</function_calls>"
_BLOCK = re.compile(re.escape(OPEN) + r"(.*?)(?:" + re.escape(CLOSE) + r"|$)", re.DOTALL)

_TOOL_INSTRUCTIONS = """

<functions>
The application you run inside can run the functions listed below for you. You cannot run
them yourself, and they are not tools available to you. To have the application run one or
more of them, end your reply with a block in exactly this form, holding a JSON list:

<function_calls>
[{{"name": "function_name", "arguments": {{"argument": "value"}}}}]
</function_calls>

The application runs them and sends you the results in its next message. Write nothing
after the block. When you need no function, just reply to the user as usual, without a block.

{listing}
</functions>"""


def _render_turn(message: Message) -> str:
    """One turn as plain text, including any tool calls or tool results it carries."""
    # wrap_untrusted: a result can never close its own tag and pose as the conversation.
    parts = [
        wrap_untrusted(
            "tool_result", r.content, name=r.name, status="error" if r.is_error else "ok"
        )
        for r in message.tool_results
    ]
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        calls = [{"name": c.name, "arguments": dict(c.arguments)} for c in message.tool_calls]
        parts.append(f"{OPEN}\n{json.dumps(calls)}\n{CLOSE}")
    return "\n".join(parts)


def render_prompt(messages: Sequence[Message]) -> str:
    """Fold a conversation into the single prompt the CLI accepts."""
    last = messages[-1]
    if len(messages) == 1:
        return _render_turn(last)
    earlier = "\n".join(f"{m.role.value.upper()}: {_render_turn(m)}" for m in messages[:-1])
    lead = (
        "Results of the functions you asked the application to run:"
        if last.tool_results
        else "Reply to the latest user message:"
    )
    return f"Conversation so far:\n{earlier}\n\n{lead}\n{_render_turn(last)}"


def _tool_listing(tools: Sequence[ToolSpec]) -> str:
    return "\n".join(
        f"- {t.name}: {t.description}\n"
        f"  Arguments (JSON schema): {json.dumps(dict(t.input_schema))}"
        for t in tools
    )


def parse_function_calls(text: str) -> tuple[str, tuple[ToolCall, ...]]:
    """Split a reply into the text for the user and the calls in its block, if any.

    A block that is not a valid JSON list of calls is left in the text: better the user sees
    it than the brain runs something the model did not clearly ask for.
    """
    match = _BLOCK.search(text)
    if match is None:
        return text, ()
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text, ()
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(c, dict) and "name" in c for c in raw):
        return text, ()
    calls = tuple(
        ToolCall(
            f"call_{uuid.uuid4().hex[:12]}",
            str(c["name"]),
            c["arguments"] if isinstance(c.get("arguments"), dict) else {},
        )
        for c in raw
    )
    return text[: match.start()].rstrip(), calls


class _HoldBack:
    """Streams text but holds back anything that may be the start of a function-call block,
    so the block itself never reaches the user."""

    def __init__(self) -> None:
        self._pending = ""
        self._blocked = False

    def feed(self, text: str) -> str:
        if self._blocked:
            return ""
        self._pending += text
        if (index := self._pending.find(OPEN)) >= 0:
            self._blocked = True
            out, self._pending = self._pending[:index], ""
            return out
        keep = max((n for n in range(1, len(OPEN)) if self._pending.endswith(OPEN[:n])), default=0)
        out = self._pending[: len(self._pending) - keep]
        self._pending = self._pending[len(out) :]
        return out

    def flush(self) -> str:
        out, self._pending = ("" if self._blocked else self._pending), ""
        return out


class ClaudeCodeProvider:
    def __init__(
        self,
        spec: ModelSpec,
        runner: CommandRunner | None = None,
        binary: str = "claude",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._spec = spec
        self._runner = runner or SubprocessRunner()
        self._binary = binary
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return f"{self._spec.provider.value}:{self._spec.model}"

    async def complete(self, request: LLMRequest) -> Completion:
        start = time.perf_counter()
        args = self._args(request, output_format="json")
        try:
            result = await self._runner.run(args, render_prompt(request.messages), self._timeout)
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(f"{self.name}: CLI '{self._binary}' not found") from exc
        except TimeoutError as exc:
            raise ProviderUnavailableError(f"{self.name}: timed out") from exc

        payload = self._parse_json(result.stdout.strip().splitlines()[-1:] or [""])
        if payload is None:
            raise ProviderUnavailableError(
                f"{self.name}: exit {result.returncode}: {result.stderr.strip()[:200]}"
            )
        completion = self._completion(payload, start)
        return _with_calls(completion) if request.offers_tools else completion

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        hold = _HoldBack() if request.offers_tools else None
        args = [*self._args(request, output_format="stream-json"), "--verbose"]
        args.append("--include-partial-messages")
        final: dict[str, Any] | None = None
        try:
            lines = self._runner.stream_lines(args, render_prompt(request.messages), self._timeout)
            async for line in lines:
                payload = self._parse_json([line])
                if payload is None:
                    continue
                if payload.get("type") == "result":
                    final = payload
                elif text := _text_delta(payload):
                    if hold is not None:
                        text = hold.feed(text)
                    if text:
                        yield TextDelta(text)
            if hold is not None and (rest := hold.flush()):
                yield TextDelta(rest)
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(f"{self.name}: CLI '{self._binary}' not found") from exc
        except TimeoutError as exc:
            raise ProviderUnavailableError(f"{self.name}: timed out") from exc

        if final is None:
            raise ProviderUnavailableError(f"{self.name}: stream ended without a result")
        completion = self._completion(final, start)
        yield _with_calls(completion) if request.offers_tools else completion

    def _args(self, request: LLMRequest, output_format: str) -> list[str]:
        system, schema = request.system, request.json_schema
        if request.offers_tools:
            system += _TOOL_INSTRUCTIONS.format(listing=_tool_listing(request.tools))
        structured = ["--json-schema", json.dumps(schema)] if schema else []
        return [*self._base_args(system, output_format), *structured]

    def _base_args(self, system: str, output_format: str) -> list[str]:
        return [
            self._binary,
            "-p",
            "--model", self._spec.model,
            "--output-format", output_format,
            "--system-prompt", system,
            "--tools", "",  # a plain model call: no built-in tools
            "--no-session-persistence",
            "--strict-mcp-config",
            "--setting-sources", "",
        ]  # fmt: skip

    @staticmethod
    def _parse_json(lines: Sequence[str]) -> dict[str, Any] | None:
        try:
            payload = json.loads(lines[0])
        except (json.JSONDecodeError, IndexError):
            return None
        return payload if isinstance(payload, dict) else None

    def _completion(self, payload: dict[str, Any], start: float) -> Completion:
        if payload.get("is_error"):
            message = str(payload.get("result") or payload.get("subtype") or "unknown error")
            status = payload.get("api_error_status")
            if isinstance(status, int):
                raise error_for_status(self.name, status, message)
            raise ProviderRequestError(f"{self.name}: {message}")

        usage = payload.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0) + cache_read + cache_write
        # With --json-schema the CLI returns the object in `structured_output` and reports
        # its internal stop reason, so a present object counts as a normal finish.
        structured = payload.get("structured_output")
        if structured is not None:
            text, stop_reason = json.dumps(structured), StopReason.END
        else:
            text = str(payload.get("result", ""))
            stop_reason = _STOP_REASONS.get(str(payload.get("stop_reason")), StopReason.OTHER)
        return Completion(
            text=text,
            spec=self._spec,
            usage=Usage(
                input_tokens, int(usage.get("output_tokens") or 0), cache_read, cache_write
            ),
            stop_reason=stop_reason,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _with_calls(completion: Completion) -> Completion:
    text, calls = parse_function_calls(completion.text)
    if not calls:
        return completion
    return replace(completion, text=text, tool_calls=calls, stop_reason=StopReason.TOOL_USE)


def _text_delta(payload: dict[str, Any]) -> str | None:
    if payload.get("type") != "stream_event":
        return None
    event = payload.get("event") or {}
    delta = event.get("delta") or {}
    if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
        return str(delta.get("text", ""))
    return None
