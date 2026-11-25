"""Access over HTTP: bearer tokens, 401s, and per-user runs and feedback."""

import time
from pathlib import Path

import jwt
import pytest
import yaml
from fastapi.testclient import TestClient

from core.access import AuditAction
from core.api.app import create_app
from core.api.context import AppContext
from core.config import ModelSpec, Provider, Settings
from core.llm import ProviderFactory, ToolCall
from core.llm.adapters.fake import FakeProvider
from tests.conftest import TEST_JWT_SECRET


@pytest.fixture
def secured_pack(pack_dir: Path) -> Path:
    (pack_dir / "policy.yaml").write_text("roles: {member: {tools: ['*']}}", encoding="utf-8")
    config = yaml.safe_load((pack_dir / "brain.yaml").read_text())
    config["access"] = {"jwt": {"secret_env": "BRAIN_JWT_SECRET", "audience": "acme"}}
    (pack_dir / "brain.yaml").write_text(yaml.safe_dump(config))
    return pack_dir


def token(sub: str, secret: str = TEST_JWT_SECRET) -> str:
    claims = {"sub": sub, "roles": ["member"], "aud": "acme", "exp": int(time.time()) + 60}
    return jwt.encode(claims, secret, algorithm="HS256")


def bearer(sub: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(sub)}"}


def ask(text: str = "hi") -> dict:
    return {"stream": False, "messages": [{"role": "user", "content": text}]}


def context(client: TestClient) -> AppContext:
    ctx: AppContext = client.app.state.context  # type: ignore[attr-defined]
    return ctx


def test_requests_without_a_valid_token_get_401_and_are_audited(secured_pack: Path) -> None:
    client = TestClient(create_app(Settings(pack=secured_pack)))

    missing = client.post("/v1/chat", json=ask())
    forged = client.post(
        "/v1/chat",
        json=ask(),
        headers={"Authorization": f"Bearer {token('u1', 'x' * 40)}"},
    )
    title = client.post("/v1/title", json=ask())

    assert (missing.status_code, forged.status_code, title.status_code) == (401, 401, 401)
    assert missing.headers["www-authenticate"] == "Bearer"
    entries = context(client).audit.recent(10)
    assert {e.action for e in entries} == {AuditAction.AUTH_FAILED}
    assert {e.target for e in entries} == {"/v1/chat", "/v1/title"}


def test_a_signed_in_user_is_answered_and_health_says_access_is_on(secured_pack: Path) -> None:
    client = TestClient(create_app(Settings(pack=secured_pack)))

    answer = client.post("/v1/chat", json=ask(), headers=bearer("u1"))

    assert answer.status_code == 200
    assert client.get("/healthz").json()["access"] == "jwt"
    trace = context(client).traces.get(answer.json()["trace_id"])
    assert trace is not None and trace.attributes["user"] == "u1"


def test_runs_and_feedback_belong_to_the_user_who_asked(secured_pack: Path) -> None:
    def fake(spec: ModelSpec) -> FakeProvider:
        return FakeProvider(spec, replies=[(ToolCall("c1", "anything", {}),), "done"])

    client = TestClient(
        create_app(Settings(pack=secured_pack), ProviderFactory({Provider.FAKE: fake}))
    )
    # No tool server, so the fake's call targets an unknown tool: still a normal answer.
    first = client.post("/v1/chat", json=ask(), headers=bearer("u1")).json()

    stolen = client.post(
        "/v1/feedback", json={"trace_id": first["trace_id"], "rating": "down"},
        headers=bearer("u2"),
    )  # fmt: skip
    own = client.post(
        "/v1/feedback", json={"trace_id": first["trace_id"], "rating": "up"},
        headers=bearer("u1"),
    )  # fmt: skip

    assert stolen.status_code == 404
    assert own.status_code == 204


def test_open_packs_stay_open(pack_dir: Path) -> None:
    client = TestClient(create_app(Settings(pack=pack_dir)))

    assert client.post("/v1/chat", json=ask()).status_code == 200
    assert client.get("/healthz").json()["access"] == "open"
