"""Layer 04, tools: discovered from MCP servers, run by the brain with limits and error handling.

The core never names a tool. Tool names, descriptions and argument schemas come from
discovery, so the same code serves any company's MCP server.
"""

from core.tools.errors import ToolError
from core.tools.factory import build_tools
from core.tools.mcp_source import McpToolSource
from core.tools.registry import ToolExecutor, ToolRegistry
from core.tools.source import ToolOutput, ToolSource

__all__ = [
    "McpToolSource",
    "ToolError",
    "ToolExecutor",
    "ToolOutput",
    "ToolRegistry",
    "ToolSource",
    "build_tools",
]
