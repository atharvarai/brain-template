import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api.app import create_app
from core.config import Provider, Settings
from core.llm import ProviderFactory, ProviderRequestError, ProviderUnavailableError
from core.llm.adapters.fake import FakeProvider
from tests.unit.llm.helpers import BrokenStreamProvider


def client(pack_dir: Path, factory: ProviderFactory | None = None) -> TestClient:
    return TestClient(create_app(Settings(pack=pack_dir), provider_factory=factory))


def failing(error: Exception) -> ProviderFactory:
    return ProviderFactory({Provider.FAKE: lambda spec: FakeProvider(spec, error=error)})


def ask(text: str = "hello", **extra: object) -> dict:
    return {"messages": [{"role": "user", "content": text}], **extra}


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_json_answer_carries_text_and_metadata(pack_dir: Path) -> None:
    response = client(pack_dir).post("/v1/chat", json=ask("hello", stream=False))

    assert response.status_code == 200
    body = response.json()
    assert body["text"].endswith("\n\nhello")  # the fake echoes the turn it was sent
    assert "(no matching documents were found)" in body["text"]
    assert body["sources"] == []
    assert body["provider"] == "fake"
    assert body["model"] == "fake-1"
    assert body["stop_reason"] == "end"
    assert body["usage"]["output_tokens"] > 0
    names = [(p["name"], p["version"]) for p in body["prompts"]]
    assert names == [("system", 2), ("persona", 1), ("rag_turn", 1)]
    assert all(len(p["checksum"]) == 12 for p in body["prompts"])


def test_streaming_is_the_default_and_ends_with_done(pack_dir: Path) -> None:
    response = client(pack_dir).post("/v1/chat", json=ask("hello world"))

    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    text = "".join(data["text"] for name, data in events if name == "delta")
    assert text.strip().endswith("hello world")
    assert [name for name, _ in events][-1] == "done"
    assert events[-1][1]["provider"] == "fake"
    assert [p["name"] for p in events[-1][1]["prompts"]] == ["system", "persona", "rag_turn"]
    assert events[-1][1]["sources"] == []
    assert "text" not in events[-1][1]


def test_the_pack_system_prompt_reaches_the_model(pack_dir: Path) -> None:
    built: list[FakeProvider] = []

    def build(spec):  # type: ignore[no-untyped-def]
        built.append(FakeProvider(spec))
        return built[-1]

    client(pack_dir, ProviderFactory({Provider.FAKE: build})).post(
        "/v1/chat", json=ask(stream=False)
    )

    sent = [request for provider in built for request in provider.requests]
    assert len(sent) == 1
    assert "You are the AI assistant for Acme." in sent[0].system  # the core skeleton
    assert "You are a test assistant." in sent[0].system  # the pack's persona


def test_bad_requests_are_rejected_before_any_model_call(pack_dir: Path) -> None:
    api = client(pack_dir)

    assert api.post("/v1/chat", json={"messages": []}).status_code == 422
    last_is_assistant = {"messages": [{"role": "assistant", "content": "hi"}]}
    assert api.post("/v1/chat", json=last_is_assistant).status_code == 422
    assert api.post("/v1/chat", json=ask("")).status_code == 422


def test_outage_returns_503_without_leaking_details(pack_dir: Path) -> None:
    api = client(pack_dir, failing(ProviderUnavailableError("secret-host:1234 refused")))

    response = api.post("/v1/chat", json=ask(stream=False))

    assert response.status_code == 503
    assert "secret-host" not in response.text


def test_rejected_request_returns_502(pack_dir: Path) -> None:
    api = client(pack_dir, failing(ProviderRequestError("invalid api key sk-123")))

    response = api.post("/v1/chat", json=ask(stream=False))

    assert response.status_code == 502
    assert "sk-123" not in response.text


def test_stream_failure_arrives_as_an_error_event(pack_dir: Path) -> None:
    error = ProviderUnavailableError("secret-host dropped the connection")
    factory = ProviderFactory({Provider.FAKE: lambda _: BrokenStreamProvider("x", 1, error)})

    events = parse_sse(client(pack_dir, factory).post("/v1/chat", json=ask()).text)

    assert [name for name, _ in events] == ["delta", "error"]
    assert "secret-host" not in events[-1][1]["message"]
