"""Adapter for the local Claude Code CLI (``claude -p``).

Lets someone with a Claude Code subscription run the brain with no API key. It is meant
for development: the CLI takes one prompt per call, so earlier turns are folded into that
prompt, and ``max_tokens`` cannot be enforced.

Tool calling is emulated. The CLI runs tools itself, inside its own loop, which would take
access checks and approvals away from the brain. So the tools are described in the system
prompt instead, and the reply is constrained by a JSON schema to either a list of tool calls
or an answer. The brain then runs the calls, exactly as it does for vendors with native tool
calling. The cost: a request that offers tools cannot stream its text.
"""

import asyncio
import json
import os
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


_TOOL_INSTRUCTIONS = """

<functions>
The application you run inside can run the functions listed below for you. You cannot run
them yourself, and they are not tools available to you. To have the application run one or
more of them, list them in the tool_calls field of your structured reply and leave answer
empty; the application runs them and sends you the results in its next message. When you
need nothing more, or no function fits, leave tool_calls empty and write your whole reply in
answer.

{listing}
</functions>"""


def _render_turn(message: Message) -> str:
    """One turn as plain text, including any tool calls or tool results it carries."""
    parts = [
        f'<tool_result name="{r.name}" status="{"error" if r.is_error else "ok"}">\n'
        f"{r.content}\n</tool_result>"
        for r in message.tool_results
    ]
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        calls = ", ".join(f"{c.name}({json.dumps(dict(c.arguments))})" for c in message.tool_calls)
        parts.append(f"(asked the application to run: {calls})")
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


def tool_reply_schema(tools: Sequence[ToolSpec]) -> dict[str, Any]:
    """Reply shape for emulated tool calling: tool calls, or an answer."""
    variants = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "arguments"],
            "properties": {"name": {"const": t.name}, "arguments": dict(t.input_schema)},
        }
        for t in tools
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_calls", "answer"],
        "properties": {
            "tool_calls": {"type": "array", "items": {"anyOf": variants}},
            "answer": {"type": "string"},
        },
    }


def _tool_listing(tools: Sequence[ToolSpec]) -> str:
    return "\n".join(
        f"- {t.name}: {t.description}\n"
        f"  Arguments (JSON schema): {json.dumps(dict(t.input_schema))}"
        for t in tools
    )


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
        return _read_tool_reply(completion) if request.offers_tools else completion

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        if request.offers_tools:  # the reply is a JSON object, so there is no text to stream
            completion = await self.complete(request)
            if completion.text:
                yield TextDelta(completion.text)
            yield completion
            return
        start = time.perf_counter()
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
                    yield TextDelta(text)
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(f"{self.name}: CLI '{self._binary}' not found") from exc
        except TimeoutError as exc:
            raise ProviderUnavailableError(f"{self.name}: timed out") from exc

        if final is None:
            raise ProviderUnavailableError(f"{self.name}: stream ended without a result")
        yield self._completion(final, start)

    def _args(self, request: LLMRequest, output_format: str) -> list[str]:
        system, schema = request.system, request.json_schema
        if request.offers_tools:
            system += _TOOL_INSTRUCTIONS.format(listing=_tool_listing(request.tools))
            schema = tool_reply_schema(request.tools)
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
        input_tokens = sum(
            int(usage.get(key) or 0)
            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
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
            usage=Usage(input_tokens, int(usage.get("output_tokens") or 0)),
            stop_reason=stop_reason,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _read_tool_reply(completion: Completion) -> Completion:
    """Turn the emulated ``{tool_calls, answer}`` object into a normal completion."""
    try:
        reply = json.loads(completion.text)
    except json.JSONDecodeError:
        return completion  # not the expected object: treat it as plain text
    if not isinstance(reply, dict):
        return completion
    calls = tuple(
        ToolCall(f"call_{uuid.uuid4().hex[:12]}", str(c["name"]), dict(c.get("arguments") or {}))
        for c in reply.get("tool_calls") or ()
        if isinstance(c, dict) and "name" in c
    )
    if calls:
        return replace(completion, text="", tool_calls=calls, stop_reason=StopReason.TOOL_USE)
    return replace(completion, text=str(reply.get("answer", "")))


def _text_delta(payload: dict[str, Any]) -> str | None:
    if payload.get("type") != "stream_event":
        return None
    event = payload.get("event") or {}
    delta = event.get("delta") or {}
    if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
        return str(delta.get("text", ""))
    return None
