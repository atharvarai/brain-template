"""The registry routes calls by name and turns every failure into a result the model can read."""

from collections.abc import Mapping, Sequence
from typing import Any

import anyio
import pytest

from core.llm import ToolCall, ToolSpec
from core.tools import ToolError, ToolInfo, ToolOutput, ToolRegistry

pytestmark = pytest.mark.anyio


def tool(name: str) -> ToolInfo:
    spec = ToolSpec(name, f"{name} tool", {"type": "object", "properties": {}})
    return ToolInfo(spec, read_only=name.startswith(("find", "get")))


class FakeSource:
    def __init__(
        self,
        name: str,
        tools: Sequence[str] = (),
        output: ToolOutput | None = None,
        open_error: Exception | None = None,
        call_error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._name = name
        self._tools = tools
        self._output = output or ToolOutput("done")
        self._open_error = open_error
        self._call_error = call_error
        self._delay = delay
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def open(self) -> Sequence[ToolInfo]:
        if self._open_error:
            raise self._open_error
        return [tool(name) for name in self._tools]

    async def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutput:
        self.calls.append((name, dict(arguments)))
        if self._delay:
            await anyio.sleep(self._delay)
        if self._call_error:
            raise self._call_error
        return self._output

    async def close(self) -> None:
        self.closed = True


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall("call_1", name, arguments)


async def test_calls_are_routed_to_the_source_that_offers_the_tool() -> None:
    orders, people = FakeSource("orders", ["find_order"]), FakeSource("people", ["find_person"])

    async with ToolRegistry([orders, people]) as registry:
        assert [s.name for s in registry.specs] == ["find_order", "find_person"]
        assert registry.is_read_only("find_order")
        result = await registry.run(call("find_person", id=7))

    assert people.calls == [("find_person", {"id": 7})]
    assert orders.calls == []
    assert (result.call_id, result.name, result.content, result.is_error) == (
        "call_1", "find_person", "done", False,
    )  # fmt: skip
    assert orders.closed and people.closed


async def test_an_unreachable_server_is_skipped_so_the_brain_still_starts() -> None:
    down = FakeSource("down", ["a"], open_error=ConnectionError("refused"))
    up = FakeSource("up", ["b"])

    async with ToolRegistry([down, up]) as registry:
        assert [s.name for s in registry.specs] == ["b"]
        assert registry.unavailable == ["down"]


async def test_two_servers_offering_the_same_tool_is_a_setup_error() -> None:
    first, second = FakeSource("one", ["lookup"]), FakeSource("two", ["lookup"])

    with pytest.raises(ToolError, match="'lookup' is offered by both 'one' and 'two'"):
        await ToolRegistry([first, second]).open()
    assert first.closed and second.closed  # nothing left connected


async def test_an_unknown_tool_name_gets_an_error_naming_the_real_tools() -> None:
    async with ToolRegistry([FakeSource("s", ["find_order"])]) as registry:
        result = await registry.run(call("delete_everything"))

    assert result.is_error
    assert "Unknown tool 'delete_everything'" in result.content
    assert "find_order" in result.content


async def test_an_error_reported_by_the_tool_reaches_the_model_as_is() -> None:
    source = FakeSource("s", ["find_order"], output=ToolOutput("no order A-7", is_error=True))

    async with ToolRegistry([source]) as registry:
        result = await registry.run(call("find_order"))

    assert (result.content, result.is_error) == ("no order A-7", True)


async def test_a_crash_is_reported_without_internal_details() -> None:
    source = FakeSource("s", ["find_order"], call_error=OSError("10.0.0.7:5432 refused"))

    async with ToolRegistry([source]) as registry:
        result = await registry.run(call("find_order"))

    assert result.is_error
    assert "10.0.0.7" not in result.content
    assert "could not be reached" in result.content


async def test_a_slow_tool_times_out() -> None:
    source = FakeSource("s", ["slow"], delay=1.0)

    async with ToolRegistry([source], timeout_seconds=0.05) as registry:
        result = await registry.run(call("slow"))

    assert result.is_error
    assert "0.05 seconds" in result.content


async def test_long_results_are_cut_with_a_note() -> None:
    source = FakeSource("s", ["dump"], output=ToolOutput("x" * 250))

    async with ToolRegistry([source], max_result_chars=100) as registry:
        result = await registry.run(call("dump"))

    assert result.content.startswith("x" * 100 + "\n")
    assert "150 more characters" in result.content
    assert not result.is_error
