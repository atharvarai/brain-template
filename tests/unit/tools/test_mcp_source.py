"""The MCP adapter against real MCP servers: in-process, and the demo pack's over stdio."""

import json
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as ServerToolError
from mcp_types import ToolAnnotations

from core.access import DevTokenIssuer
from core.config import ToolsConfig, load_pack
from core.llm import ToolCall
from core.tools import McpToolSource, ToolRegistry, build_tools
from examples.edtech.mcp_server.server import build_server
from tests.conftest import EDTECH_PACK

pytestmark = pytest.mark.anyio


def demo_server() -> MCPServer:
    server = MCPServer("demo")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool()
    def find(code: str) -> str:
        """Find a record by code."""
        raise ServerToolError(f"no record {code}")

    return server


async def test_tools_are_discovered_with_their_schemas() -> None:
    source = McpToolSource("demo", demo_server())
    specs = await source.open()
    try:
        by_name = {info.spec.name: info for info in specs}
        assert set(by_name) == {"add", "find"}
        assert by_name["add"].spec.description == "Add two numbers."
        assert by_name["add"].spec.input_schema["required"] == ["a", "b"]
        assert by_name["add"].read_only  # declared with readOnlyHint
        assert not by_name["find"].read_only  # no hint: treated as changing data
    finally:
        await source.close()


async def test_results_and_tool_errors_come_back_as_outputs() -> None:
    source = McpToolSource("demo", demo_server())
    await source.open()
    try:
        assert (await source.call("add", {"a": 2, "b": 3})).content == "5"
        failed = await source.call("find", {"code": "Z9"})
        assert failed.is_error
        assert "no record Z9" in failed.content
        invalid = await source.call("add", {"a": "two"})
        assert invalid.is_error  # argument validation happens on the server
    finally:
        await source.close()


async def test_calling_before_open_is_a_programming_error() -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        await McpToolSource("demo", demo_server()).call("add", {})


# --- the Skilvanta demo server --------------------------------------------------------------


def token(handle: str = "riya") -> str:
    pack = load_pack(EDTECH_PACK)
    assert pack.config.access is not None
    return DevTokenIssuer(pack.config.access, pack.root).issue(handle)


async def run(
    tool: str, as_user: str | None = "riya", **arguments: object
) -> tuple[dict | str, bool]:
    credentials = token(as_user) if as_user else None
    async with ToolRegistry([McpToolSource("skilvanta", build_server())]) as registry:
        result = await registry.run(ToolCall("c1", tool, arguments), credentials)
    try:
        return json.loads(result.content), result.is_error
    except json.JSONDecodeError:
        return result.content, result.is_error


async def test_demo_enrollments_match_the_fee_rules_in_the_docs() -> None:
    body, is_error = await run("get_my_enrollments")

    assert not is_error
    assert isinstance(body, dict)
    courses = {e["course"]: e for e in body["enrollments"]}
    assert set(courses) == {"Backend Engineering with Python", "DSA and Interview Prep"}
    # payments doc: Rs 60,000 over 6 months is Rs 10,000 per month
    assert "Rs 10,000" in courses["Backend Engineering with Python"]["payment"]


async def test_demo_progress_accepts_a_course_name_and_refuses_other_courses() -> None:
    body, is_error = await run("get_course_progress", course="backend engineering")
    assert not is_error
    assert isinstance(body, dict)
    assert body["modules_completed"] == 9

    message, is_error = await run("get_course_progress", course="DSML")
    assert is_error
    assert "not enrolled in Data Science and Machine Learning" in str(message)


async def test_demo_slots_stay_inside_the_window_asked_for() -> None:
    body, _ = await run("list_mentor_slots", days_ahead=7)
    assert isinstance(body, dict)
    assert body["mentor"] == "Arjun Desai"
    assert body["open_slots_ist"] == ["2026-11-27T19:00", "2026-11-29T11:00", "2026-12-02T20:00"]
    assert body["sessions_left_this_month"] == 1

    _, is_error = await run("list_mentor_slots", days_ahead=60)
    assert is_error


async def test_the_pack_starts_its_server_over_stdio() -> None:
    """End to end through a real child process, as `make run` does."""
    config = ToolsConfig.model_validate(
        {"servers": {"skilvanta": {"transport": "stdio", "pass_env": ["BRAIN_JWT_SECRET"],
                                   "command": ["python", "mcp_server/server.py"]}}}
    )  # fmt: skip

    async with build_tools(config, EDTECH_PACK) as registry:
        names = {s.name for s in registry.specs}
        assert names == {
            "get_my_enrollments", "get_course_progress", "list_mentor_slots",
            "get_mentee_progress", "book_mentor_session", "request_refund",
        }  # fmt: skip
        writes = {name for name in names if not registry.is_read_only(name)}
        assert writes == {"book_mentor_session", "request_refund"}
        result = await registry.run(ToolCall("c1", "get_my_enrollments", {}), token("riya"))
        anonymous = await registry.run(ToolCall("c2", "get_my_enrollments", {}))

    assert "Riya Kulkarni" in result.content  # the token crossed the process boundary
    assert anonymous.is_error and "Not signed in" in anonymous.content


async def test_a_server_that_cannot_start_leaves_the_brain_without_its_tools(
    tmp_path: Path,
) -> None:
    config = ToolsConfig.model_validate(
        {"servers": {"broken": {"transport": "stdio", "command": ["python", "missing.py"]}}}
    )

    async with build_tools(config, tmp_path) as registry:
        assert registry.specs == ()
        assert registry.unavailable == ["broken"]


CRASHY = '''
import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("crashy")


@server.tool()
def pid() -> int:
    """The server's process id."""
    return os.getpid()


@server.tool()
def crash() -> str:
    """Exit at once, like a server that died."""
    os._exit(1)


server.run()
'''


async def test_a_dead_server_is_reconnected_on_a_later_call(tmp_path: Path) -> None:
    (tmp_path / "crashy.py").write_text(CRASHY, encoding="utf-8")
    target = StdioServerParameters(command=sys.executable, args=["crashy.py"], cwd=tmp_path)
    source = McpToolSource("crashy", target, reconnect_wait_seconds=10)
    await source.open()
    try:
        first_pid = (await source.call("pid", {})).content
        with pytest.raises(Exception):  # noqa: B017 - whatever the transport raises
            await source.call("crash", {})
        second_pid = (await source.call("pid", {})).content  # waits for the reconnect
    finally:
        await source.close()

    assert first_pid != second_pid  # a fresh server process


async def test_demo_booking_uses_a_session_and_takes_the_slot() -> None:
    riya = token("riya")
    async with ToolRegistry([McpToolSource("skilvanta", build_server())]) as registry:
        booked = await registry.run(
            ToolCall("c1", "book_mentor_session", {"slot_ist": "2026-11-27T19:00", "agenda": "A7"}),
            riya,
        )
        twice = await registry.run(
            ToolCall("c2", "book_mentor_session", {"slot_ist": "2026-11-29T11:00", "agenda": "x"}),
            riya,
        )

    body = json.loads(booked.content)
    assert body["sessions_left_that_month"] == 0
    assert body["booking_id"].startswith("BK-")
    assert twice.is_error and "No sessions left in 2026-11" in twice.content


async def test_demo_refund_follows_the_policy_document() -> None:
    body, is_error = await run("request_refund", course="DSA", reason="no time")

    assert not is_error and isinstance(body, dict)
    assert body["estimated_refund"] == "Rs 0"
    assert body["policy_rule"] == "no refund: requested after day 21"


def test_refund_rules_match_the_worked_example_in_the_docs() -> None:
    from examples.edtech.mcp_server.server import _refund_estimate

    assert _refund_estimate(85_000, 5, 1)[0] == 85_000
    assert _refund_estimate(85_000, 15, 0)[0] == 42_500
    assert _refund_estimate(85_000, 25, 0)[0] == 0


@pytest.mark.parametrize(
    ("as_user", "learner", "allowed"),
    [
        ("arjun", "L-1042", True),  # his own mentee
        ("arjun", "L-1077", False),  # someone else's mentee
        ("sana", "L-1077", True),  # support sees every learner
        ("riya", "L-1077", False),  # a learner is not staff
    ],
)
async def test_the_demo_server_applies_its_own_rules_to_each_user(
    as_user: str, learner: str, allowed: bool
) -> None:
    course = "BEPY" if learner == "L-1042" else "DSML"
    _, is_error = await run("get_mentee_progress", as_user, learner_id=learner, course=course)
    assert is_error is not allowed


async def test_the_demo_server_refuses_missing_and_forged_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message, is_error = await run("get_my_enrollments", None)
    assert is_error and "Not signed in" in str(message)

    forged = token("riya")
    monkeypatch.setenv("BRAIN_JWT_SECRET", "a-different-secret-the-server-does-not-know-xx")
    async with ToolRegistry([McpToolSource("skilvanta", build_server())]) as registry:
        result = await registry.run(ToolCall("c1", "get_my_enrollments", {}), forged)
    assert result.is_error and "not accepted" in result.content
