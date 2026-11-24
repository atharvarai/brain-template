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
    (tmp_path / "prompts" / "persona.v1.md").write_text(
        "You are a test assistant.", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only."""
    return "asyncio"


DOCS = {
    "returns.md": (
        "---\ntitle: Returns Policy\nlabels: []\n---\n# Returns Policy\n\n"
        "## Window\nAnvils can be returned within 30 days of delivery.\n\n"
        "## Shipping\nReturn shipping is free for damaged anvils.\n"
    ),
    "pricing.md": "# Pricing\n\nThe standard anvil costs 120 dollars.\n",
    "staff/margins.md": (
        "---\ntitle: Margins\nlabels: [internal]\n---\n# Margins\n\n"
        "The anvil margin is 62 percent. Returns cost us 9 dollars each.\n"
    ),
}


@pytest.fixture
def rag_pack_dir(pack_dir: Path) -> Path:
    """The Acme pack with a small document set and the model-free hashing embedder."""
    for name, text in DOCS.items():
        path = pack_dir / "docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    config = yaml.safe_load((pack_dir / "brain.yaml").read_text())
    config["rag"] = {"docs": "docs", "top_k": 3, "embedder": {"kind": "hashing"}}
    (pack_dir / "brain.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return pack_dir
