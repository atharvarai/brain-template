"""Every example pack must load and hang together: config, prompts, docs, users, eval cases
and the tools its cases name. A new pack under examples/ is checked with no test changes."""

import importlib
from pathlib import Path

import anyio
import pytest

from core.access import RolePolicy, load_dev_users
from core.api.app import build_context
from core.config import Settings, load_pack
from core.evals import load_cases
from core.llm import default_factory
from tests.conftest import REPO_ROOT

PACKS = sorted(p.parent for p in (REPO_ROOT / "examples").glob("*/brain.yaml"))


@pytest.fixture(params=PACKS, ids=[p.name for p in PACKS])
def pack_root(request: pytest.FixtureRequest) -> Path:
    root: Path = request.param
    return root


def test_there_are_at_least_two_example_packs() -> None:
    assert {p.name for p in PACKS} >= {"edtech", "rides"}


def test_the_pack_builds_a_complete_brain(pack_root: Path) -> None:
    context = build_context(Settings(pack=pack_root), default_factory())
    assert context.pack.config.name


def test_eval_cases_name_real_sections_users_and_tools(pack_root: Path) -> None:
    pack = load_pack(pack_root)
    cases = load_cases(pack.eval_cases)
    docs_root = pack_root / "docs"
    docs = {p.relative_to(docs_root).as_posix() for p in docs_root.rglob("*.md")}
    sections = {
        f"{name}#{line.lstrip('#').strip()}"
        for name in docs
        for line in (docs_root / name).read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    }
    users = load_dev_users(pack_root / pack.config.access.dev_users) if pack.config.access else {}
    server = importlib.import_module(f"examples.{pack_root.name}.mcp_server.server")
    tools = {t.name for t in anyio.run(server.build_server().list_tools)}

    assert len(cases) >= 15
    for case in cases:
        for expected in (*case.expect.retrieves, *case.expect.cites):
            assert expected in (sections if "#" in expected else docs), (case.id, expected)
        assert case.user is None or case.user in users, (case.id, case.user)
        for tool in (*(case.expect.calls or ()), *(case.expect.asks_approval or ())):
            assert tool in tools, (case.id, tool)


def test_the_policy_only_grants_tools_the_server_offers(pack_root: Path) -> None:
    pack = load_pack(pack_root)
    if pack.config.access is None:
        pytest.skip("open pack")
    policy = RolePolicy.load(pack_root / pack.config.access.policy)
    server = importlib.import_module(f"examples.{pack_root.name}.mcp_server.server")
    offered = {t.name for t in anyio.run(server.build_server().list_tools)}
    granted = {tool for rules in policy._roles.values() for tool in rules.tools}
    assert granted - {"*"} <= offered
