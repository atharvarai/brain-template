import json
from collections.abc import AsyncIterator, Sequence

import pytest

from core.config import Provider
from core.llm import (
    Message,
    ProviderRequestError,
    ProviderUnavailableError,
    Role,
    StopReason,
)
from core.llm.adapters.claude_code import (
    ClaudeCodeProvider,
    CommandResult,
    SubprocessRunner,
    render_prompt,
)
from tests.unit.llm.helpers import collect, request, spec

pytestmark = pytest.mark.anyio

RESULT = {
    "type": "result",
    "is_error": False,
    "result": "pong",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 2, "cache_creation_input_tokens": 600, "output_tokens": 4},
}


class StubRunner:
    def __init__(
        self,
        result: CommandResult | None = None,
        lines: Sequence[str] = (),
        error: Exception | None = None,
    ) -> None:
        self._result, self._lines, self._error = result, lines, error
        self.calls: list[tuple[list[str], str]] = []

    async def run(self, args: Sequence[str], stdin: str, timeout: float) -> CommandResult:
        self.calls.append((list(args), stdin))
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result

    async def stream_lines(
        self, args: Sequence[str], stdin: str, timeout: float
    ) -> AsyncIterator[str]:
        self.calls.append((list(args), stdin))
        if self._error:
            raise self._error
        for line in self._lines:
            yield line


def provider(runner: StubRunner) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(spec(Provider.CLAUDE_CODE, "haiku"), runner=runner)


def ok(payload: dict) -> CommandResult:
    return CommandResult(0, json.dumps(payload), "")


async def test_complete_parses_the_cli_json() -> None:
    runner = StubRunner(ok(RESULT))

    completion = await provider(runner).complete(request("ping", system="be terse"))

    assert completion.text == "pong"
    assert completion.stop_reason is StopReason.END
    assert completion.usage.input_tokens == 602  # cached input counts as input
    assert completion.usage.output_tokens == 4
    args, stdin = runner.calls[0]
    assert stdin == "ping"
    assert args[args.index("--model") + 1] == "haiku"
    assert args[args.index("--system-prompt") + 1] == "be terse"
    assert args[args.index("--tools") + 1] == ""  # built-in tools are switched off


async def test_stream_yields_deltas_then_the_final_completion() -> None:
    def delta(text: str) -> str:
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        return json.dumps({"type": "stream_event", "event": event})

    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        delta("po"),
        "not json at all",
        delta("ng"),
        json.dumps(RESULT),
    ]

    text, final = await collect(provider(StubRunner(lines=lines)).stream(request()))

    assert text == "pong"
    assert final.text == "pong"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"is_error": True, "api_error_status": 529, "result": "x"}, ProviderUnavailableError),
        ({"is_error": True, "api_error_status": 429, "result": "x"}, ProviderUnavailableError),
        ({"is_error": True, "api_error_status": 400, "result": "x"}, ProviderRequestError),
        ({"is_error": True, "result": "Invalid model"}, ProviderRequestError),
    ],
)
async def test_cli_errors_map_onto_the_taxonomy(payload: dict, expected: type) -> None:
    with pytest.raises(expected):
        await provider(StubRunner(ok(payload))).complete(request())


@pytest.mark.parametrize("error", [FileNotFoundError("claude"), TimeoutError()])
async def test_missing_cli_or_timeout_means_unavailable(error: Exception) -> None:
    with pytest.raises(ProviderUnavailableError):
        await provider(StubRunner(error=error)).complete(request())
    with pytest.raises(ProviderUnavailableError):
        await collect(provider(StubRunner(error=error)).stream(request()))


async def test_garbage_output_means_unavailable() -> None:
    with pytest.raises(ProviderUnavailableError, match="exit 1"):
        await provider(StubRunner(CommandResult(1, "", "crashed"))).complete(request())
    with pytest.raises(ProviderUnavailableError, match="without a result"):
        await collect(provider(StubRunner(lines=["{}"])).stream(request()))


def test_render_prompt_folds_history_into_one_prompt() -> None:
    history = (
        Message(Role.USER, "My name is Asha"),
        Message(Role.ASSISTANT, "Hi Asha"),
        Message(Role.USER, "What is my name?"),
    )

    prompt = render_prompt(history)

    assert "USER: My name is Asha" in prompt
    assert "ASSISTANT: Hi Asha" in prompt
    assert prompt.endswith("What is my name?")
    assert render_prompt(history[:1]) == "My name is Asha"


async def test_subprocess_runner_hides_the_api_key_and_pipes_stdin(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    script = (
        "import os,sys; print(sys.stdin.read() + '|' + os.environ.get('ANTHROPIC_API_KEY','unset'))"
    )
    runner = SubprocessRunner()

    result = await runner.run(["python3", "-c", script], "hello", timeout=10)
    lines = [line async for line in runner.stream_lines(["python3", "-c", script], "hi", 10)]

    assert result.stdout.strip() == "hello|unset"
    assert lines == ["hi|unset\n"]


async def test_subprocess_runner_times_out() -> None:
    with pytest.raises(TimeoutError):
        await SubprocessRunner().run(["python3", "-c", "import time; time.sleep(5)"], "", 0.2)
