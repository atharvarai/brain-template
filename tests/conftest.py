"""Shared fixtures."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EDTECH_PACK = REPO_ROOT / "examples" / "edtech"


def _tiers(provider: str, model: str) -> dict[str, list[dict[str, str]]]:
    return {tier: [{"provider": provider, "model": model}] for tier in ("small", "mid", "large")}


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    """A minimal valid pack for a made-up company, unrelated to any shipped example."""
    config = {
        "name": "Acme",
        "models": {
            "active": "test",
            "profiles": {"test": _tiers("fake", "fake-1"), "alt": _tiers("ollama", "tiny")},
        },
    }
    (tmp_path / "brain.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("You are a test assistant.", encoding="utf-8")
    return tmp_path
