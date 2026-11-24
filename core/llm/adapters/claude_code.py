"""Adapter for the local Claude Code CLI (``claude -p``).

Lets someone with a Claude Code subscription run the brain with no API key. It is meant
for development: the CLI takes one prompt per call, so earlier turns are folded into that
prompt, and ``max_tokens`` cannot be enforced.
"""

import asyncio
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
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


def render_prompt(messages: Sequence[Message]) -> str:
    """Fold a conversation into the single prompt the CLI accepts."""
    if len(messages) == 1:
        return messages[0].content
    earlier = "\n".join(f"{m.role.value.upper()}: {m.content}" for m in messages[:-1])
    return (
        f"Conversation so far:\n{earlier}\n\n"
        f"Reply to the latest user message:\n{messages[-1].content}"
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
        return self._completion(payload, start)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
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
        return [
            self._binary,
            "-p",
            "--model", self._spec.model,
            "--output-format", output_format,
            "--system-prompt", request.system,
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
        return Completion(
            text=str(payload.get("result", "")),
            spec=self._spec,
            usage=Usage(input_tokens, int(usage.get("output_tokens") or 0)),
            stop_reason=_STOP_REASONS.get(str(payload.get("stop_reason")), StopReason.OTHER),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _text_delta(payload: dict[str, Any]) -> str | None:
    if payload.get("type") != "stream_event":
        return None
    event = payload.get("event") or {}
    delta = event.get("delta") or {}
    if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
        return str(delta.get("text", ""))
    return None
