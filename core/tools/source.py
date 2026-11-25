"""The port a tool provider implements. MCP is the one adapter; tests use in-memory fakes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from core.llm import ToolSpec


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """What a tool returned. ``is_error`` means the tool ran and reported a failure."""

    content: str
    is_error: bool = False


class ToolSource(Protocol):
    """A set of tools behind one connection, for example one MCP server."""

    @property
    def name(self) -> str: ...

    async def open(self) -> Sequence[ToolSpec]:
        """Connect and discover the tools on offer."""
        ...

    async def call(self, tool: str, arguments: Mapping[str, Any]) -> ToolOutput:
        """Run one tool. Raises if the source cannot be reached; tool failures are outputs."""
        ...

    async def close(self) -> None: ...
