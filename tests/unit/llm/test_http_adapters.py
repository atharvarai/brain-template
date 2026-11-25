"""Anthropic, OpenAI and Ollama adapters, run through the real SDKs with mocked HTTP."""

import json
from collections.abc import Callable

import anthropic
import httpx2
import openai
import pytest

from core.config import Provider
from core.llm import (
    LLMProvider,
    LLMRequest,
    Message,
    ProviderRequestError,
    ProviderUnavailableError,
    Role,
    StopReason,
)
from core.llm.adapters.anthropic_adapter import AnthropicProvider
from core.llm.adapters.ollama import OllamaProvider
from core.llm.adapters.openai_adapter import OpenAIProvider
from tests.unit.llm.helpers import collect, request, spec

pytestmark = pytest.mark.anyio

Handler = Callable[[httpx2.Request], httpx2.Response]


def http(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def anthropic_provider(handler: Handler) -> LLMProvider:
    client = anthropic.AsyncAnthropic(api_key="test", http_client=http(handler), max_retries=0)
    return AnthropicProvider(spec(Provider.ANTHROPIC, "claude-x"), client=client)


def openai_provider(handler: Handler) -> LLMProvider:
    client = openai.AsyncOpenAI(api_key="test", http_client=http(handler), max_retries=0)
    return OpenAIProvider(spec(Provider.OPENAI, "gpt-x"), client=client)


def ollama_provider(handler: Handler) -> LLMProvider:
    return OllamaProvider(spec(Provider.OLLAMA, "llama-x"), client=http(handler))


def sse(*events: dict | str) -> httpx2.Response:
    body = "".join(
        f"data: {e}\n\n" if isinstance(e, str) else f"event: {e['type']}\ndata: {json.dumps(e)}\n\n"
        for e in events
    )
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


# --- canned vendor responses -------------------------------------------------------------

ANTHROPIC_MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
    "content": [{"type": "text", "text": "Hello there"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 5,
              "cache_creation_input_tokens": 0},
}  # fmt: skip


def anthropic_stream(_: httpx2.Request) -> httpx2.Response:
    start = {**ANTHROPIC_MESSAGE, "content": [], "stop_reason": None}
    start["usage"] = {"input_tokens": 10, "output_tokens": 1}
    block = {"type": "text", "text": ""}
    return sse(
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )  # fmt: skip


OPENAI_COMPLETION = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-x",
    "choices": [{"index": 0, "finish_reason": "stop", "logprobs": None,
                 "message": {"role": "assistant", "content": "Hello there"}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
}  # fmt: skip


def openai_stream(_: httpx2.Request) -> httpx2.Response:
    def chunk(delta: dict, finish: str | None = None) -> str:
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        return json.dumps({"id": "c1", "object": "chat.completion.chunk", "created": 1,
                           "model": "gpt-x", "choices": [choice]})  # fmt: skip

    usage = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-x",
             "choices": [],
             "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}}  # fmt: skip
    return sse(
        chunk({"role": "assistant", "content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="length"),
        json.dumps(usage),
        "[DONE]",
    )


def ollama_stream(_: httpx2.Request) -> httpx2.Response:
    lines = [
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length",
         "prompt_eval_count": 7, "eval_count": 2},
    ]  # fmt: skip
    return httpx2.Response(200, content="\n".join(json.dumps(line) for line in lines))


OLLAMA_COMPLETION = {
    "message": {"role": "assistant", "content": "Hello there"},
    "done": True, "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 2,
}  # fmt: skip

PROVIDERS = [
    pytest.param(anthropic_provider, ANTHROPIC_MESSAGE, anthropic_stream, id="anthropic"),
    pytest.param(openai_provider, OPENAI_COMPLETION, openai_stream, id="openai"),
    pytest.param(ollama_provider, OLLAMA_COMPLETION, ollama_stream, id="ollama"),
]


# --- behaviour every adapter must share ----------------------------------------------------


@pytest.mark.parametrize(("build", "body", "_stream"), PROVIDERS)
async def test_complete_maps_the_vendor_response(build, body, _stream) -> None:
    sent: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=body)

    completion = await build(handler).complete(request("hi", system="be brief"))

    assert completion.text == "Hello there"
    assert completion.stop_reason is StopReason.END
    assert completion.usage.output_tokens == 2
    assert completion.usage.input_tokens > 0
    assert "be brief" in json.dumps(sent[0])  # the system prompt was sent
    assert sent[0]["messages"][-1] == {"role": "user", "content": "hi"}


@pytest.mark.parametrize(("build", "_body", "stream"), PROVIDERS)
async def test_stream_yields_deltas_then_a_completion(build, _body, stream) -> None:
    text, final = await collect(build(stream).stream(request()))

    assert text == "Hello"
    assert final.text == "Hello"
    assert final.stop_reason is StopReason.MAX_TOKENS
    assert final.usage.output_tokens == 2


@pytest.mark.parametrize(("build", "_body", "_stream"), PROVIDERS)
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, ProviderUnavailableError),
        (500, ProviderUnavailableError),
        (529, ProviderUnavailableError),
        (400, ProviderRequestError),
        (401, ProviderRequestError),
    ],
)
async def test_http_errors_map_onto_the_taxonomy(build, _body, _stream, status, expected) -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"type": "x", "message": "nope"}})

    with pytest.raises(expected):
        await build(handler).complete(request())
    with pytest.raises(expected):
        await collect(build(handler).stream(request()))


@pytest.mark.parametrize(("build", "_body", "_stream"), PROVIDERS)
async def test_network_failure_means_unavailable(build, _body, _stream) -> None:
    def handler(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=req)

    with pytest.raises(ProviderUnavailableError):
        await build(handler).complete(request())
    with pytest.raises(ProviderUnavailableError):
        await collect(build(handler).stream(request()))


# --- vendor-specific details --------------------------------------------------------------


async def test_anthropic_counts_cached_input_and_sends_system_separately() -> None:
    sent: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=ANTHROPIC_MESSAGE)

    completion = await anthropic_provider(handler).complete(request(system="be brief"))

    assert completion.usage.input_tokens == 15  # 10 fresh + 5 read from cache
    assert completion.usage.cache_read_tokens == 5
    # The system prompt is marked as a cache breakpoint; a plain answer's messages are not.
    assert sent[0]["system"] == [
        {"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}
    ]
    assert sent[0]["messages"][-1]["content"] == "hello"
    assert sent[0]["max_tokens"] == 2048


async def test_openai_puts_the_system_prompt_first_and_caps_output() -> None:
    sent: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=OPENAI_COMPLETION)

    await openai_provider(handler).complete(request(system="be brief"))

    assert sent[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent[0]["max_completion_tokens"] == 2048


async def test_ollama_uses_the_configured_host(monkeypatch) -> None:
    urls: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        urls.append(str(req.url))
        return httpx2.Response(200, json=OLLAMA_COMPLETION)

    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434/")
    await OllamaProvider(spec(Provider.OLLAMA, "llama-x"), client=http(handler)).complete(request())

    assert urls == ["http://gpu-box:11434/api/chat"]


@pytest.mark.parametrize(
    ("build", "body", "locate"),
    [
        pytest.param(
            anthropic_provider, ANTHROPIC_MESSAGE,
            lambda sent: sent["output_config"]["format"], id="anthropic",
        ),
        pytest.param(
            openai_provider, OPENAI_COMPLETION,
            lambda sent: sent["response_format"]["json_schema"], id="openai",
        ),
        pytest.param(ollama_provider, OLLAMA_COMPLETION, lambda sent: sent, id="ollama"),
    ],
)  # fmt: skip
async def test_json_schema_uses_the_vendor_native_feature(build, body, locate) -> None:
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    sent: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(req.content))
        return httpx2.Response(200, json=body)

    provider = build(handler)
    await provider.complete(request())
    structured = LLMRequest("s", (Message(Role.USER, "x"),), json_schema=schema)
    await provider.complete(structured)

    plain, with_schema = sent
    assert not {"output_config", "response_format", "format"} & plain.keys()
    carrier = locate(with_schema)
    assert carrier.get("schema", carrier.get("format")) == schema
