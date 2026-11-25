"""Builds the tool registry from a pack's ``tools`` section."""

import sys
from pathlib import Path

from mcp.client.stdio import StdioServerParameters

from core.config import HttpServerConfig, McpServerConfig, ToolsConfig
from core.tools.mcp_source import McpTarget, McpToolSource
from core.tools.registry import ToolRegistry


def build_tools(config: ToolsConfig, pack_root: Path) -> ToolRegistry:
    """Not connected yet: call ``open()`` (or use ``async with``) before running tools."""
    sources = [
        McpToolSource(name, _target(server, pack_root), config.timeout_seconds)
        for name, server in config.servers.items()
    ]
    return ToolRegistry(sources, config.timeout_seconds, config.max_result_chars)


def _target(server: McpServerConfig, pack_root: Path) -> McpTarget:
    if isinstance(server, HttpServerConfig):
        return server.url
    command, *args = server.command
    if command == "python":  # the brain's own interpreter, so the pack needs no set-up
        command = sys.executable
    return StdioServerParameters(
        command=command, args=args, env=dict(server.env) or None, cwd=pack_root
    )
