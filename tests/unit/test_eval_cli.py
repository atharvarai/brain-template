"""`eval` and `traces` commands, end to end with the fake model and hashing embedder."""

import json
from pathlib import Path

import pytest
import yaml

from core import cli
from core.observability import SqliteTraceStore

CASES = [
    {
        "id": "window",
        "question": "how long do I have to return an anvil",
        "tags": ["returns"],
        "expect": {"retrieves": ["returns.md"], "contains_any": ["30 days"]},
    },
    {"id": "price", "question": "anvil price", "expect": {"retrieves": ["pricing.md"]}},
    {"id": "tone", "question": "hi", "expect": {"rubric": "Is polite."}},
]


@pytest.fixture
def eval_pack(rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (rag_pack_dir / "evals").mkdir()
    (rag_pack_dir / "evals" / "cases.jsonl").write_text(
        "\n".join(json.dumps(case) for case in CASES), encoding="utf-8"
    )
    monkeypatch.setenv("BRAIN_PACK", str(rag_pack_dir))
    assert cli.main(["ingest"]) == 0
    return rag_pack_dir


def test_retrieval_mode_needs_no_model_and_passes(
    eval_pack: Path, isolated_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["eval", "--retrieval"]) == 0
    out = capsys.readouterr().out
    assert "retrieval mode" in out
    assert "PASS  window" in out
    assert "skipped (no retrieval checks): 1 cases" in out
    assert "Pass rate 2/2 = 100.0%" in out
    assert list((isolated_data_dir / eval_pack.name / "evals").glob("retrieval-*.json"))


def test_answer_mode_grades_with_the_judge_and_fails_below_the_bar(
    eval_pack: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The fake model echoes the documents it was given, and the fake judge always says no.
    assert cli.main(["eval"]) == 1
    out = capsys.readouterr().out
    assert "PASS  window" in out
    assert "FAIL  tone" in out
    assert "BELOW THRESHOLD" in out

    assert cli.main(["eval", "--min-pass-rate", "0.6", "--no-save"]) == 0
    assert "Previous run" in capsys.readouterr().out


def test_filters_select_cases_by_tag_and_id(
    eval_pack: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["eval", "--retrieval", "--tag", "returns", "--no-save"]) == 0
    assert "1 cases" in capsys.readouterr().out
    assert cli.main(["eval", "--case", "nope"]) == 1
    assert "no cases match" in capsys.readouterr().out


def test_traces_lists_recent_requests_and_shows_one_in_full(
    eval_pack: Path, isolated_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["traces"]) == 0
    assert "no traces yet" in capsys.readouterr().out

    cli.main(["eval", "--no-save"])
    capsys.readouterr()
    assert cli.main(["traces", "--operation", "judge"]) == 0
    out = capsys.readouterr().out
    assert " judge " in out
    assert "1 traces, 0 errors" in out

    assert cli.main(["traces", "-n", "50"]) == 0
    assert "4 traces" in capsys.readouterr().out  # three eval answers and one judge call

    store = SqliteTraceStore(isolated_data_dir / eval_pack.name / "traces.sqlite3")
    [latest] = store.recent(1, operation="eval")
    assert cli.main(["traces", "--id", latest.trace_id]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["operation"] == "eval"
    assert [span["name"] for span in shown["spans"]] == ["retrieve", "generate"]
    assert cli.main(["traces", "--id", "nope"]) == 1


def test_a_pack_without_cases_is_a_clear_error(
    rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("BRAIN_PACK", str(rag_pack_dir))
    assert cli.main(["eval", "--retrieval"]) == 1
    assert "no eval cases" in capsys.readouterr().err


SERVER = '''
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("acme")


@server.tool()
def find_order(order_id: str) -> str:
    """Find an order by id."""
    if order_id == "missing":
        raise ToolError("no such order")
    return f"Order {order_id}: shipped"


server.run()
'''


def test_tools_lists_discovered_tools_and_calls_one(
    pack_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (pack_dir / "acme_tools.py").write_text(SERVER, encoding="utf-8")
    config = yaml.safe_load((pack_dir / "brain.yaml").read_text())
    config["tools"] = {
        "servers": {"acme": {"transport": "stdio", "command": ["python", "acme_tools.py"]}}
    }
    (pack_dir / "brain.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("BRAIN_PACK", str(pack_dir))

    assert cli.main(["tools"]) == 0
    assert "find_order (changes data): Find an order by id." in capsys.readouterr().out

    assert cli.main(["tools", "--call", "find_order", "--args", '{"order_id": "A-7"}']) == 0
    assert "Order A-7: shipped" in capsys.readouterr().out

    assert cli.main(["tools", "--call", "find_order", "--args", '{"order_id": "missing"}']) == 1
    assert "ERROR:" in capsys.readouterr().out
    assert cli.main(["tools", "--call", "find_order", "--args", "[1]"]) == 1


def test_tools_says_so_when_the_pack_has_no_servers(
    pack_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAIN_PACK", str(pack_dir))
    assert cli.main(["tools"]) == 1
