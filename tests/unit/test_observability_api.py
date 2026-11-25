"""Every request leaves a trace; feedback joins to it by trace_id."""

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.api.context import AppContext
from core.config import ModelSpec, Provider, Settings
from core.llm import ProviderFactory, ProviderUnavailableError
from core.llm.adapters.fake import FakeProvider
from core.observability import TraceStatus
from tests.unit.llm.helpers import BrokenStreamProvider


def make_client(pack_dir: Path, factory: ProviderFactory | None = None) -> TestClient:
    return TestClient(create_app(Settings(pack=pack_dir), provider_factory=factory))


def context(client: TestClient) -> AppContext:
    ctx: AppContext = client.app.state.context  # type: ignore[attr-defined]
    return ctx


def ask(text: str = "hello", **extra: object) -> dict:
    return {"messages": [{"role": "user", "content": text}], **extra}


def edit_pack(pack_dir: Path, **changes: object) -> None:
    path = pack_dir / "brain.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(changes)
    path.write_text(yaml.safe_dump(config))


def test_a_json_answer_returns_its_trace_id_and_the_trace_explains_it(pack_dir: Path) -> None:
    client = make_client(pack_dir)
    body = client.post("/v1/chat", json=ask("hello", stream=False)).json()

    trace = context(client).traces.get(body["trace_id"])
    assert trace is not None
    assert trace.operation == "chat"
    assert trace.status is TraceStatus.OK
    assert (trace.provider, trace.model) == ("fake", "fake-1")
    assert [s.name for s in trace.spans] == ["retrieve", "generate"]
    assert [p.split("#")[0] for p in trace.prompts] == ["system@v5", "persona@v1", "rag_turn@v1"]
    assert trace.output_tokens == body["usage"]["output_tokens"]
    assert trace.cost_usd is None  # the test pack declares no prices
    assert trace.input is None and trace.output is None  # content capture is off by default
    assert trace.attributes == {
        "pack": "Acme", "profile": "test", "user": "anonymous", "steps": 1, "tool_calls": 0,
        "tier": "mid", "route": "fixed",
    }  # fmt: skip


def test_prices_and_content_capture_come_from_the_pack(pack_dir: Path) -> None:
    config = yaml.safe_load((pack_dir / "brain.yaml").read_text())
    config["models"]["prices"] = {"fake-1": {"input": 1.0, "output": 2.0}}
    edit_pack(pack_dir, models=config["models"], observability={"capture_content": True})
    client = make_client(pack_dir)
    body = client.post("/v1/chat", json=ask("what is the fee?", stream=False)).json()

    trace = context(client).traces.get(body["trace_id"])
    assert trace is not None
    usage = body["usage"]
    assert trace.cost_usd == (usage["input_tokens"] * 1 + usage["output_tokens"] * 2) / 1e6
    assert trace.input == "what is the fee?"
    assert trace.output == body["text"]


def test_a_stream_records_time_to_first_token_and_ends_with_the_trace_id(
    pack_dir: Path,
) -> None:
    client = make_client(pack_dir)
    response = client.post("/v1/chat", json=ask("hello world"))
    done = response.text.strip().split("\n\n")[-1]
    trace_id = done.split('"trace_id": "')[1].split('"')[0]

    trace = context(client).traces.get(trace_id)
    assert trace is not None
    assert [e.name for e in trace.events] == ["first_token"]


def test_a_failed_stream_is_traced_as_an_error(pack_dir: Path) -> None:
    error = ProviderUnavailableError("connection reset")
    factory = ProviderFactory({Provider.FAKE: lambda _: BrokenStreamProvider("x", 1, error)})
    client = make_client(pack_dir, factory)
    response = client.post("/v1/chat", json=ask("hello"))
    assert "event: error" in response.text

    [trace] = context(client).traces.recent(5)
    assert trace.status is TraceStatus.ERROR
    assert trace.error is not None


def test_failover_shows_up_as_an_event_on_the_trace(pack_dir: Path) -> None:
    chain = [{"provider": "fake", "model": "down"}, {"provider": "fake", "model": "up"}]
    tiers = {tier: chain for tier in ("small", "mid", "large")}
    edit_pack(pack_dir, models={"active": "test", "profiles": {"test": tiers}})

    def build(spec: ModelSpec) -> FakeProvider:
        if spec.model == "down":
            return FakeProvider(spec, error=ProviderUnavailableError("fake:down overloaded"))
        return FakeProvider(spec)

    client = make_client(pack_dir, ProviderFactory({Provider.FAKE: build}))
    body = client.post("/v1/chat", json=ask("hello", stream=False)).json()

    trace = context(client).traces.get(body["trace_id"])
    assert trace is not None
    assert trace.model == "up"
    [event] = trace.events
    assert event.name == "provider_failed"
    assert event.attributes["provider"] == "fake:down"


def test_feedback_is_stored_against_a_real_trace(pack_dir: Path) -> None:
    client = make_client(pack_dir)
    trace_id = client.post("/v1/chat", json=ask("hello", stream=False)).json()["trace_id"]

    ok = client.post("/v1/feedback", json={"trace_id": trace_id, "rating": "down", "comment": "x"})
    assert ok.status_code == 204
    stored = context(client).traces.feedback_for([trace_id])[trace_id]
    assert (stored.rating.value, stored.comment) == ("down", "x")

    unknown = client.post("/v1/feedback", json={"trace_id": "nope", "rating": "up"})
    assert unknown.status_code == 404
    invalid = client.post("/v1/feedback", json={"trace_id": trace_id, "rating": "meh"})
    assert invalid.status_code == 422


def test_titles_are_traced_too(pack_dir: Path) -> None:
    client = make_client(pack_dir)
    body = client.post("/v1/title", json=ask("how do refunds work")).json()

    trace = context(client).traces.get(body["trace_id"])
    assert trace is not None
    assert trace.operation == "title"
    assert trace.prompts[0].startswith("title@v1#")
