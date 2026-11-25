"""One tool round trip in the pipeline: offer tools, run the calls, ask again for the answer."""

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.config import ModelSpec, ModelTier, Provider, Settings, load_pack
from core.llm import (
    ModelGateway,
    ProviderFactory,
    Role,
    StopReason,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)
from core.llm.adapters.fake import FakeProvider, Reply
from core.llm.types import Message
from core.observability import InMemoryTraceStore, Tracer
from core.pipeline import Answer, Pipeline, ToolActivity
from core.prompts import ChainedPromptStore, FilePromptStore, build_system_prompt, builtin_store
from core.rag import NullRetriever
from tests.unit.test_chat_api import parse_sse

pytestmark = pytest.mark.anyio

ORDER = ToolSpec("find_order", "Find an order.", {"type": "object", "properties": {}})
STOCK = ToolSpec("check_stock", "Check stock.", {"type": "object", "properties": {}})


class FakeTools:
    def __init__(self, failing: Sequence[str] = ()) -> None:
        self.specs = (ORDER, STOCK)
        self._failing = set(failing)
        self.ran: list[ToolCall] = []

    async def run(self, call: ToolCall) -> ToolResult:
        self.ran.append(call)
        if call.name in self._failing:
            return ToolResult(call.id, call.name, "service down", is_error=True)
        return ToolResult(call.id, call.name, f"{call.name} result")


def calls(*names: str) -> tuple[ToolCall, ...]:
    return tuple(ToolCall(f"c{i}", name, {}) for i, name in enumerate(names))


def build(
    pack_dir: Path, replies: Sequence[Reply], tools: FakeTools, max_tool_calls: int = 4
) -> tuple[Pipeline, FakeProvider, InMemoryTraceStore]:
    pack = load_pack(pack_dir)
    prompts = ChainedPromptStore([FilePromptStore(pack.prompts_dir), builtin_store()])
    provider = FakeProvider(ModelSpec(provider=Provider.FAKE, model="m"), replies=replies)
    store = InMemoryTraceStore()
    pipeline = Pipeline(
        ModelGateway({tier: provider for tier in ModelTier}),
        build_system_prompt(prompts, pack.config),
        NullRetriever(),
        prompts.get("rag_turn"),
        Tracer(store),
        tools=tools,
        max_tool_calls=max_tool_calls,
    )
    return pipeline, provider, store


def ask(text: str = "where is my order?") -> list[Message]:
    return [Message(Role.USER, text)]


async def test_the_model_is_offered_the_tools_and_answers_without_them(pack_dir: Path) -> None:
    pipeline, provider, store = build(pack_dir, ["No tool needed."], FakeTools())

    answer = await pipeline.answer(ask())

    assert provider.requests[0].tools == (ORDER, STOCK)
    assert len(provider.requests) == 1
    assert (answer.completion.text, answer.tools) == ("No tool needed.", ())
    [trace] = store.recent(1)
    assert [s.name for s in trace.spans] == ["retrieve", "generate"]


async def test_one_round_trip_runs_the_calls_and_asks_again(pack_dir: Path) -> None:
    tools = FakeTools()
    pipeline, provider, store = build(
        pack_dir, [calls("find_order", "check_stock"), "It ships tomorrow."], tools
    )

    answer = await pipeline.answer(ask())

    assert [c.name for c in tools.ran] == ["find_order", "check_stock"]
    second = provider.requests[1]
    assert second.tool_choice is ToolChoice.NONE  # one round only: the agent does more
    assistant, results = second.messages[-2:]
    assert [c.name for c in assistant.tool_calls] == ["find_order", "check_stock"]
    assert [r.content for r in results.tool_results] == ["find_order result", "check_stock result"]
    assert answer.completion.text == "It ships tomorrow."
    assert answer.completion.stop_reason is StopReason.END
    assert [a.name for a in answer.tools] == ["find_order", "check_stock"]
    # Usage covers both model calls, and so does the trace.
    first_usage = 2  # the fake counts one output token per tool call
    assert answer.completion.usage.output_tokens == first_usage + 3
    [trace] = store.recent(1)
    assert [s.name for s in trace.spans] == ["retrieve", "generate", "tool", "tool", "generate"]
    assert trace.output_tokens == answer.completion.usage.output_tokens


async def test_a_failed_tool_is_reported_to_the_model_and_marked_on_the_answer(
    pack_dir: Path,
) -> None:
    pipeline, provider, store = build(
        pack_dir, [calls("find_order"), "I could not look that up."], FakeTools(["find_order"])
    )

    answer = await pipeline.answer(ask())

    [result] = provider.requests[1].messages[-1].tool_results
    assert (result.content, result.is_error) == ("service down", True)
    assert answer.tools[0].ok is False
    [trace] = store.recent(1)
    tool_span = next(s for s in trace.spans if s.name == "tool")
    assert tool_span.attributes == {"tool": "find_order", "ok": False}


async def test_calls_over_the_limit_get_an_error_result_instead_of_running(
    pack_dir: Path,
) -> None:
    tools = FakeTools()
    pipeline, provider, _ = build(
        pack_dir, [calls("find_order", "check_stock", "find_order"), "done"], tools,
        max_tool_calls=2,
    )  # fmt: skip

    answer = await pipeline.answer(ask())

    assert len(tools.ran) == 2
    results = provider.requests[1].messages[-1].tool_results
    assert len(results) == 3  # every call the model made gets a result
    assert results[2].is_error and "at most 2 tool calls" in results[2].content
    assert len(answer.tools) == 2


async def test_streaming_reports_each_tool_then_streams_the_final_answer(pack_dir: Path) -> None:
    pipeline, _, _ = build(pack_dir, [calls("find_order"), "It ships tomorrow."], FakeTools())

    events = [event async for event in pipeline.stream(ask())]

    kinds = [type(event) for event in events]
    assert kinds[0] is ToolActivity
    assert kinds[-1] is Answer
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text.strip() == "It ships tomorrow."
    answer = events[-1]
    assert isinstance(answer, Answer)
    assert answer.tools == (events[0],)


# --- through the API, with a real MCP server over stdio -------------------------------------

SERVER = '''
from mcp.server.mcpserver import MCPServer

server = MCPServer("acme")


@server.tool()
def find_order(order_id: str) -> str:
    """Find an order by id."""
    return f"Order {order_id}: shipped"


server.run()
'''


def tool_pack(pack_dir: Path, command: list[str]) -> Path:
    (pack_dir / "acme_tools.py").write_text(SERVER, encoding="utf-8")
    path = pack_dir / "brain.yaml"
    config = yaml.safe_load(path.read_text())
    config["tools"] = {"servers": {"acme": {"transport": "stdio", "command": command}}}
    path.write_text(yaml.safe_dump(config))
    return pack_dir


def test_chat_runs_a_discovered_tool_and_streams_a_tool_event(pack_dir: Path) -> None:
    tool_pack(pack_dir, ["python", "acme_tools.py"])
    call = ToolCall("c1", "find_order", {"order_id": "A-7"})
    providers: list[FakeProvider] = []

    def fake(spec: ModelSpec) -> FakeProvider:
        providers.append(FakeProvider(spec, replies=[(call,), "Your order has shipped."]))
        return providers[-1]

    app = create_app(Settings(pack=pack_dir), ProviderFactory({Provider.FAKE: fake}))
    with TestClient(app) as client:  # `with` runs the lifespan, which connects the tools
        health = client.get("/healthz").json()
        assert (health["status"], health["tools"]) == ("ok", ["find_order"])

        events = parse_sse(client.post("/v1/chat", json={"messages": [
            {"role": "user", "content": "where is A-7?"}]}).text)  # fmt: skip

    names = [name for name, _ in events]
    assert names[0] == "tool" and names[-1] == "done"
    assert events[0][1]["name"] == "find_order" and events[0][1]["ok"] is True
    assert events[-1][1]["tools"][0]["name"] == "find_order"
    sent = next(p for p in providers if p.requests).requests[1]
    [result] = sent.messages[-1].tool_results
    assert result.content == "Order A-7: shipped"


def test_health_is_degraded_when_a_tool_server_is_down(pack_dir: Path) -> None:
    tool_pack(pack_dir, ["python", "does_not_exist.py"])

    with TestClient(create_app(Settings(pack=pack_dir))) as client:
        health = client.get("/healthz").json()
        answer = client.post("/v1/chat", json={"stream": False, "messages": [
            {"role": "user", "content": "hello"}]})  # fmt: skip

    assert health["status"] == "degraded"
    assert health["tool_servers_down"] == ["acme"]
    assert answer.status_code == 200  # the brain still answers, just without tools
