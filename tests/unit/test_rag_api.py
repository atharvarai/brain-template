"""Retrieval wired into the chat endpoint and the CLI, with the fake model."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core import cli
from core.api.app import create_app
from core.config import ModelSpec, Provider, Settings
from core.llm import ProviderFactory
from core.llm.adapters.fake import FakeProvider
from core.rag import RagError


def settings(pack: Path) -> Settings:
    return Settings(pack=pack, data_dir=pack / ".brain")


def ingest(pack: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setenv("BRAIN_PACK", str(pack))
    monkeypatch.setenv("BRAIN_DATA_DIR", str(pack / ".brain"))
    return cli.main(["ingest", *args])


def ask(client: TestClient, text: str) -> dict:
    body = {"stream": False, "messages": [{"role": "user", "content": text}]}
    return client.post("/v1/chat", json=body).json()


def test_answer_is_grounded_in_retrieved_public_documents(
    rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ingest(rag_pack_dir, monkeypatch) == 0
    built: list[FakeProvider] = []

    def build(spec: ModelSpec) -> FakeProvider:
        built.append(FakeProvider(spec, reply="Thirty days [1]."))
        return built[-1]

    client = TestClient(create_app(settings(rag_pack_dir), ProviderFactory({Provider.FAKE: build})))
    body = ask(client, "how many days to return an anvil")

    sources = body["sources"]
    assert sources[0]["source"] == "returns.md" and sources[0]["section"] == "Window"
    assert [s["cited"] for s in sources] == [True] + [False] * (len(sources) - 1)
    assert len(sources) <= 3  # top_k from brain.yaml
    assert all(not s["source"].startswith("staff/") for s in sources)  # restricted doc hidden

    (sent,) = [request for provider in built for request in provider.requests]
    turn = sent.messages[-1].content
    assert "returned within 30 days" in turn and "<documents>" in turn
    assert turn.endswith("how many days to return an anvil")
    assert "62 percent" not in turn  # the internal margin never reaches the model
    assert "30 days" not in sent.system  # documents go in the turn: the system prompt stays stable


def test_only_the_latest_turn_carries_documents(
    rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest(rag_pack_dir, monkeypatch)
    built: list[FakeProvider] = []

    def build(spec: ModelSpec) -> FakeProvider:
        built.append(FakeProvider(spec))
        return built[-1]

    client = TestClient(create_app(settings(rag_pack_dir), ProviderFactory({Provider.FAKE: build})))
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what does the anvil cost"},
    ]
    client.post("/v1/chat", json={"stream": False, "messages": history})

    (sent,) = [request for provider in built for request in provider.requests]
    assert [m.content for m in sent.messages[:2]] == ["hi", "hello"]
    assert "120 dollars" in sent.messages[2].content


def test_streaming_reports_sources_in_the_done_event(
    rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest(rag_pack_dir, monkeypatch)
    client = TestClient(create_app(settings(rag_pack_dir)))

    response = client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "anvil returns"}]}
    )

    done = json.loads(response.text.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert done["sources"] and done["sources"][0]["index"] == 1


def test_service_starts_and_answers_before_any_ingest(rag_pack_dir: Path) -> None:
    client = TestClient(create_app(settings(rag_pack_dir)))

    body = ask(client, "anything")

    assert body["sources"] == []
    assert "(no matching documents were found)" in body["text"]


def test_missing_docs_folder_fails_at_startup(rag_pack_dir: Path) -> None:
    for path in sorted((rag_pack_dir / "docs").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (rag_pack_dir / "docs").rmdir()

    with pytest.raises(RagError, match="documents folder not found"):
        create_app(settings(rag_pack_dir))


def test_cli_ingest_and_search(
    rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest(rag_pack_dir, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "3 added" in out and "4 chunks" in out

    assert cli.main(["search", "anvil margin"]) == 0
    assert "staff/margins.md" not in capsys.readouterr().out  # public only by default

    assert cli.main(["search", "anvil margin", "--all-labels", "-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "staff/margins.md" in out and "labels=['internal']" in out

    assert ingest(rag_pack_dir, monkeypatch, "--rebuild") == 0
    assert "3 added" in capsys.readouterr().out


def test_cli_explains_a_pack_without_rag(
    pack_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest(pack_dir, monkeypatch) == 1
    assert "no 'rag' section" in capsys.readouterr().out
