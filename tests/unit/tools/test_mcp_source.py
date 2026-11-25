"""The MCP adapter against real MCP servers: in-process, and the demo pack's over stdio."""

import json
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as ServerToolError

from core.config import ToolsConfig
from core.llm import ToolCall
from core.tools import McpToolSource, ToolRegistry, build_tools
from examples.edtech.mcp_server.server import build_server
from tests.conftest import EDTECH_PACK

pytestmark = pytest.mark.anyio


def demo_server() -> MCPServer:
    server = MCPServer("demo")

    @server.tool()
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
        by_name = {spec.name: spec for spec in specs}
        assert set(by_name) == {"add", "find"}
        assert by_name["add"].description == "Add two numbers."
        assert by_name["add"].input_schema["required"] == ["a", "b"]
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


async def run(tool: str, **arguments: object) -> tuple[dict | str, bool]:
    async with ToolRegistry([McpToolSource("skilvanta", build_server())]) as registry:
        result = await registry.run(ToolCall("c1", tool, arguments))
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
        {"servers": {"skilvanta": {"transport": "stdio",
                                   "command": ["python", "mcp_server/server.py"]}}}
    )  # fmt: skip

    async with build_tools(config, EDTECH_PACK) as registry:
        assert {s.name for s in registry.specs} == {
            "get_my_enrollments", "get_course_progress", "list_mentor_slots",
        }  # fmt: skip
        result = await registry.run(ToolCall("c1", "get_my_enrollments", {}))

    assert "Riya Kulkarni" in result.content


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
