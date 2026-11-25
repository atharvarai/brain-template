"""Tool calling through every adapter: the same neutral types in, each vendor's shape out."""

import json

import httpx2
import pytest

from core.config import Provider
from core.llm import (
    LLMRequest,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)
from core.llm.adapters.claude_code import ClaudeCodeProvider, parse_function_calls, render_prompt
from core.llm.adapters.fake import FakeProvider
from tests.unit.llm.helpers import collect, spec
from tests.unit.llm.test_claude_code import RESULT, StubRunner, ok
from tests.unit.llm.test_http_adapters import (
    ANTHROPIC_MESSAGE,
    OLLAMA_COMPLETION,
    OPENAI_COMPLETION,
    anthropic_provider,
    ollama_provider,
    openai_provider,
    sse,
)

pytestmark = pytest.mark.anyio

LOOKUP = ToolSpec(
    "lookup_order",
    "Find an order by id.",
    {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
)
CALL = ToolCall("call_1", "lookup_order", {"order_id": "A-7"})
RESULT_OK = ToolResult("call_1", "lookup_order", '{"status": "shipped"}')


def offer(text: str = "where is order A-7?") -> LLMRequest:
    return LLMRequest("be brief", (Message(Role.USER, text),), tools=(LOOKUP,))


def follow_up() -> LLMRequest:
    """The second call of a round trip: the call, its result, and no more calls allowed."""
    return LLMRequest(
        "be brief",
        (
            Message(Role.USER, "where is order A-7?"),
            Message(Role.ASSISTANT, "Checking.", tool_calls=(CALL,)),
            Message(Role.USER, "", tool_results=(RESULT_OK,)),
        ),
        tools=(LOOKUP,),
        tool_choice=ToolChoice.NONE,
    )


# --- neutral types --------------------------------------------------------------------------


def test_tool_calls_and_results_sit_on_the_right_roles() -> None:
    with pytest.raises(ValueError, match="assistant"):
        Message(Role.USER, "x", tool_calls=(CALL,))
    with pytest.raises(ValueError, match="user"):
        Message(Role.ASSISTANT, "x", tool_results=(RESULT_OK,))


def test_a_request_rejects_duplicate_tools_and_tools_with_structured_output() -> None:
    with pytest.raises(ValueError, match="unique"):
        LLMRequest("s", (Message(Role.USER, "x"),), tools=(LOOKUP, LOOKUP))
    with pytest.raises(ValueError, match="structured output"):
        LLMRequest("s", (Message(Role.USER, "x"),), tools=(LOOKUP,), json_schema={})


def test_offers_tools_only_when_calls_are_allowed() -> None:
    assert offer().offers_tools
    assert not follow_up().offers_tools
    assert not LLMRequest("s", (Message(Role.USER, "x"),)).offers_tools


async def test_the_fake_provider_can_script_a_tool_call_then_an_answer() -> None:
    fake = FakeProvider(spec(), replies=[(CALL,), "It has shipped."])

    first = await fake.complete(offer())
    second = await fake.complete(follow_up())

    assert (first.stop_reason, first.tool_calls, first.text) == (StopReason.TOOL_USE, (CALL,), "")
    assert (second.stop_reason, second.text) == (StopReason.END, "It has shipped.")


# --- Anthropic ------------------------------------------------------------------------------


async def test_anthropic_sends_tools_and_reads_tool_use_blocks() -> None:
    sent: list[dict] = []
    reply = {
        **ANTHROPIC_MESSAGE,
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup_order",
             "input": {"order_id": "A-7"}},
        ],
    }  # fmt: skip

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=reply)

    completion = await anthropic_provider(handler).complete(offer())

    assert sent[0]["tools"][0]["name"] == "lookup_order"
    assert sent[0]["tools"][0]["input_schema"] == LOOKUP.input_schema
    assert sent[0]["tool_choice"] == {"type": "auto"}
    # A step that may call tools caches tools and the conversation for the next step.
    assert sent[0]["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert sent[0]["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert completion.stop_reason is StopReason.TOOL_USE
    assert completion.text == "Checking."
    assert completion.tool_calls == (ToolCall("toolu_1", "lookup_order", {"order_id": "A-7"}),)


async def test_anthropic_maps_the_round_trip_history_onto_content_blocks() -> None:
    sent: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=ANTHROPIC_MESSAGE)

    await anthropic_provider(handler).complete(follow_up())

    _, assistant, results = sent[0]["messages"]
    assert assistant["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "call_1", "name": "lookup_order", "input": {"order_id": "A-7"}},
    ]
    assert results["role"] == "user"
    assert results["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": '{"status": "shipped"}',
         "is_error": False},
    ]  # fmt: skip
    assert sent[0]["tool_choice"] == {"type": "none"}  # tools still defined, calls not allowed


# --- OpenAI ---------------------------------------------------------------------------------


async def test_openai_sends_functions_and_parses_json_arguments() -> None:
    sent: list[dict] = []
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_9", "type": "function",
             "function": {"name": "lookup_order", "arguments": '{"order_id": "A-7"}'}},
            {"id": "call_10", "type": "function",
             "function": {"name": "lookup_order", "arguments": "{not json"}},
        ],
    }  # fmt: skip
    reply = {
        **OPENAI_COMPLETION,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "logprobs": None,
                     "message": message}],
    }  # fmt: skip

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=reply)

    completion = await openai_provider(handler).complete(offer())

    assert sent[0]["tools"][0]["function"]["parameters"] == LOOKUP.input_schema
    assert sent[0]["tool_choice"] == "auto"
    assert completion.stop_reason is StopReason.TOOL_USE
    # Bad JSON from the model becomes no arguments; the tool's validation will say why.
    assert [c.arguments for c in completion.tool_calls] == [{"order_id": "A-7"}, {}]


async def test_openai_maps_results_onto_tool_messages() -> None:
    sent: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=OPENAI_COMPLETION)

    await openai_provider(handler).complete(follow_up())

    _, _, assistant, tool = sent[0]["messages"]
    assert assistant["tool_calls"][0]["function"] == {
        "name": "lookup_order",
        "arguments": '{"order_id": "A-7"}',
    }
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "shipped"}'}
    assert sent[0]["tool_choice"] == "none"


async def test_openai_assembles_tool_calls_streamed_in_pieces() -> None:
    def chunk(delta: dict, finish: str | None = None) -> str:
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        return json.dumps({"id": "c1", "object": "chat.completion.chunk", "created": 1,
                           "model": "gpt-x", "choices": [choice]})  # fmt: skip

    def piece(**fields: object) -> dict:
        return {"tool_calls": [{"index": 0, **fields}]}

    def handler(_: httpx2.Request) -> httpx2.Response:
        return sse(
            chunk(piece(id="call_5", type="function", function={"name": "lookup_order"})),
            chunk(piece(function={"arguments": '{"order_'})),
            chunk(piece(function={"arguments": 'id": "A-7"}'})),
            chunk({}, finish="tool_calls"),
            "[DONE]",
        )

    text, final = await collect(openai_provider(handler).stream(offer()))

    assert text == ""
    assert final.tool_calls == (ToolCall("call_5", "lookup_order", {"order_id": "A-7"}),)
    assert final.stop_reason is StopReason.TOOL_USE


# --- Ollama ---------------------------------------------------------------------------------


async def test_ollama_numbers_calls_and_drops_tools_when_calls_are_not_allowed() -> None:
    sent: list[dict] = []
    reply = {
        **OLLAMA_COMPLETION,
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "lookup_order", "arguments": {"order_id": "A-7"}}},
        ]},
    }  # fmt: skip

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=reply if len(sent) == 1 else OLLAMA_COMPLETION)

    provider = ollama_provider(handler)
    first = await provider.complete(offer())
    await provider.complete(follow_up())

    assert sent[0]["tools"][0]["function"]["name"] == "lookup_order"
    assert first.stop_reason is StopReason.TOOL_USE  # Ollama itself says "stop"
    assert first.tool_calls == (ToolCall("call_0", "lookup_order", {"order_id": "A-7"}),)
    assert "tools" not in sent[1]
    assert sent[1]["messages"][-1] == {
        "role": "tool",
        "tool_name": "lookup_order",
        "content": '{"status": "shipped"}',
    }


# --- Claude Code CLI (emulated) -------------------------------------------------------------

BLOCK = (
    '<function_calls>\n[{"name": "lookup_order", "arguments": {"order_id": "A-7"}}]\n'
    "</function_calls>"
)


def cli_text(text: str) -> dict:
    return {**RESULT, "result": text}


def claude_code(runner: StubRunner) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(spec(Provider.CLAUDE_CODE), runner=runner)


async def test_claude_code_describes_tools_and_parses_the_call_block() -> None:
    runner = StubRunner(ok(cli_text("Let me check.\n" + BLOCK)))

    completion = await claude_code(runner).complete(offer())

    args, _ = runner.calls[0]
    system = args[args.index("--system-prompt") + 1]
    assert system.startswith("be brief")
    assert "lookup_order: Find an order by id." in system
    assert "--json-schema" not in args  # plain text: answers stay full and can stream
    assert args[args.index("--tools") + 1] == ""  # the CLI's own tools stay off
    assert completion.stop_reason is StopReason.TOOL_USE
    assert completion.text == "Let me check."
    [call] = completion.tool_calls
    assert (call.name, call.arguments) == ("lookup_order", {"order_id": "A-7"})
    assert call.id.startswith("call_")


async def test_claude_code_passes_a_plain_answer_through() -> None:
    completion = await claude_code(StubRunner(ok(cli_text("Hello")))).complete(offer("hi"))

    assert (completion.text, completion.tool_calls) == ("Hello", ())
    assert completion.stop_reason is StopReason.END


@pytest.mark.parametrize(
    "text",
    [
        "<function_calls>\nnot json\n</function_calls>",
        '<function_calls>\n[{"arguments": {}}]\n</function_calls>',
    ],
)
def test_a_malformed_block_is_left_as_text_and_runs_nothing(text: str) -> None:
    assert parse_function_calls(text) == (text, ())


async def test_claude_code_streams_text_but_never_the_call_block() -> None:
    def delta(text: str) -> str:
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        return json.dumps({"type": "stream_event", "event": event})

    pieces = ["Let me ", "check. <func", "tion_calls>\n[", '{"name": "lookup_order", ',
              '"arguments": {"order_id": "A-7"}}]\n</function_calls>']  # fmt: skip
    lines = [delta(p) for p in pieces] + [json.dumps(cli_text("Let me check. " + BLOCK))]

    text, final = await collect(claude_code(StubRunner(lines=lines)).stream(offer()))

    assert text == "Let me check. "
    assert final.tool_calls[0].name == "lookup_order"
    assert final.text == "Let me check."


async def test_claude_code_streams_ordinary_answers_when_tools_are_offered() -> None:
    def delta(text: str) -> str:
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        return json.dumps({"type": "stream_event", "event": event})

    lines = [delta("A <b>tag</b> "), delta("and <"), delta(" more"), json.dumps(cli_text("x"))]

    text, _ = await collect(claude_code(StubRunner(lines=lines)).stream(offer()))

    assert text == "A <b>tag</b> and < more"  # held-back text is released, not lost


async def test_claude_code_sends_a_plain_call_once_tools_may_not_be_called() -> None:
    runner = StubRunner(ok(RESULT))

    completion = await claude_code(runner).complete(follow_up())

    args, stdin = runner.calls[0]
    assert "<functions>" not in args[args.index("--system-prompt") + 1]
    assert completion.text == "pong"
    assert "Results of the functions you asked the application to run:" in stdin
    assert '<tool_result name="lookup_order" status="ok">' in stdin
    assert "ASSISTANT: Checking.\n<function_calls>" in stdin  # its own earlier call


def test_render_prompt_marks_failed_results() -> None:
    failed = ToolResult("call_1", "lookup_order", "no such order", is_error=True)
    prompt = render_prompt(
        (
            Message(Role.USER, "where is A-7?"),
            Message(Role.ASSISTANT, "", tool_calls=(CALL,)),
            Message(Role.USER, "", tool_results=(failed,)),
        )
    )

    assert '"name": "lookup_order"' in prompt
    assert '<tool_result name="lookup_order" status="error">\nno such order' in prompt
