"""Approvals over HTTP, with a real MCP server over stdio: chat pauses, decisions resume."""

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.config import ModelSpec, Provider, Settings
from core.llm import ProviderFactory, ToolCall
from core.llm.adapters.fake import FakeProvider, Reply
from tests.unit.test_chat_api import parse_sse

SERVER = '''
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

server = MCPServer("acme")
CANCELLED = []


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def find_order(order_id: str) -> str:
    """Find an order by id."""
    return f"Order {order_id}: shipped" + (" (cancelled)" if order_id in CANCELLED else "")


@server.tool()
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    CANCELLED.append(order_id)
    return f"Order {order_id} cancelled"


server.run()
'''

FIND = ToolCall("c1", "find_order", {"order_id": "A-7"})
CANCEL = ToolCall("c2", "cancel_order", {"order_id": "A-7"})


def make_client(pack_dir: Path, replies: list[Reply]) -> TestClient:
    (pack_dir / "acme_tools.py").write_text(SERVER, encoding="utf-8")
    path = pack_dir / "brain.yaml"
    config = yaml.safe_load(path.read_text())
    config["tools"] = {
        "servers": {"acme": {"transport": "stdio", "command": ["python", "acme_tools.py"]}}
    }
    path.write_text(yaml.safe_dump(config))

    def fake(spec: ModelSpec) -> FakeProvider:
        return FakeProvider(spec, replies=replies)

    return TestClient(create_app(Settings(pack=pack_dir), ProviderFactory({Provider.FAKE: fake})))


def chat(client: TestClient, text: str, stream: bool = False) -> dict:
    body = {"stream": stream, "messages": [{"role": "user", "content": text}]}
    response = client.post("/v1/chat", json=body)
    assert response.status_code == 200
    return response.json()


def test_a_read_tool_runs_and_the_answer_lists_it(pack_dir: Path) -> None:
    with make_client(pack_dir, [(FIND,), "It has shipped."]) as client:
        health = client.get("/healthz").json()
        events = parse_sse(
            client.post(
                "/v1/chat", json={"messages": [{"role": "user", "content": "where is A-7?"}]}
            ).text
        )

    assert health["tools"] == ["find_order", "cancel_order"]
    names = [name for name, _ in events]
    assert names[0] == "tool" and names[-1] == "done"
    done = events[-1][1]
    assert done["status"] == "complete"
    assert [(t["name"], t["ok"], t["step"]) for t in done["tools"]] == [("find_order", True, 1)]


def test_a_write_tool_waits_for_approval_then_runs(pack_dir: Path) -> None:
    with make_client(pack_dir, [(CANCEL,), "Cancelled.", (FIND,), "Shows cancelled."]) as client:
        paused = chat(client, "cancel A-7")
        assert paused["status"] == "awaiting_approval"
        assert paused["pending"] == [
            {"call_id": "c2", "tool": "cancel_order", "arguments": {"order_id": "A-7"}}
        ]
        run_id = paused["run_id"]

        decided = client.post(
            f"/v1/runs/{run_id}/decisions",
            json={"stream": False, "decisions": [{"call_id": "c2", "approve": True}]},
        )
        again = client.post(
            f"/v1/runs/{run_id}/decisions",
            json={"stream": False, "decisions": [{"call_id": "c2", "approve": True}]},
        )
        check = chat(client, "and now?")  # the server really changed state

    assert decided.status_code == 200
    body = decided.json()
    assert (body["status"], body["text"]) == ("complete", "Cancelled.")
    assert [t["name"] for t in body["tools"]] == ["cancel_order"]
    assert again.status_code == 409
    assert check["text"] == "Shows cancelled."


def test_decisions_stream_and_bad_requests_get_clear_errors(pack_dir: Path) -> None:
    with make_client(pack_dir, [(CANCEL,), "Left as is."]) as client:
        events = parse_sse(
            client.post(
                "/v1/chat", json={"messages": [{"role": "user", "content": "cancel A-7"}]}
            ).text
        )
        [(_name, approval)] = [(n, d) for n, d in events if n == "approval"]
        run_id = approval["run_id"]
        assert events[-1][1]["status"] == "awaiting_approval"

        unknown = client.post(
            "/v1/runs/nope/decisions", json={"decisions": [{"call_id": "c2", "approve": True}]}
        )
        mismatched = client.post(
            f"/v1/runs/{run_id}/decisions",
            json={"decisions": [{"call_id": "zz", "approve": True}]},
        )
        declined = client.post(
            f"/v1/runs/{run_id}/decisions",
            json={"decisions": [{"call_id": "c2", "approve": False, "note": "not now"}]},
        )

    assert unknown.status_code == 404
    assert mismatched.status_code == 422
    assert declined.status_code == 200
    streamed = parse_sse(declined.text)
    assert streamed[-1][0] == "done"
    assert streamed[-1][1]["status"] == "complete"
    assert streamed[-1][1]["tools"] == []  # declined: nothing ran


def test_health_is_degraded_when_a_tool_server_is_down(pack_dir: Path) -> None:
    make_client(pack_dir, ["hello"])  # writes the server and the tools config
    config = yaml.safe_load((pack_dir / "brain.yaml").read_text())
    config["tools"]["servers"]["acme"]["command"] = ["python", "does_not_exist.py"]
    (pack_dir / "brain.yaml").write_text(yaml.safe_dump(config))

    with TestClient(create_app(Settings(pack=pack_dir))) as client:
        health = client.get("/healthz").json()
        answer = client.post(
            "/v1/chat", json={"stream": False, "messages": [{"role": "user", "content": "hi"}]}
        )

    assert health["status"] == "degraded"
    assert health["tool_servers_down"] == ["acme"]
    assert answer.status_code == 200  # the brain still answers, just without tools
