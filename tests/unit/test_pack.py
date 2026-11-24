from pathlib import Path

import pytest
import yaml

from core.config import ModelTier, PackError, Provider, load_pack
from tests.conftest import EDTECH_PACK


def _rewrite(pack_dir: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    path = pack_dir / "brain.yaml"
    config = yaml.safe_load(path.read_text())
    mutate(config)
    path.write_text(yaml.safe_dump(config))


def test_loads_a_valid_pack(pack_dir: Path) -> None:
    pack = load_pack(pack_dir)

    assert pack.config.name == "Acme"
    chain = pack.config.models.chain(ModelTier.SMALL)
    assert chain[0].provider is Provider.FAKE
    assert pack.system_prompt_path.read_text() == "You are a test assistant."


def test_chain_can_be_read_from_a_non_active_profile(pack_dir: Path) -> None:
    models = load_pack(pack_dir).config.models

    assert models.chain(ModelTier.LARGE, profile="alt")[0].provider is Provider.OLLAMA
    with pytest.raises(KeyError, match="unknown model profile"):
        models.chain(ModelTier.LARGE, profile="nope")


def test_shipped_edtech_pack_is_valid() -> None:
    pack = load_pack(EDTECH_PACK)

    for profile in pack.config.models.profiles:
        for tier in ModelTier:
            assert pack.config.models.chain(tier, profile)


def test_missing_folder_or_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="folder not found"):
        load_pack(tmp_path / "nope")
    with pytest.raises(PackError, match=r"brain\.yaml not found"):
        load_pack(tmp_path)


def test_profile_missing_a_tier_is_rejected(pack_dir: Path) -> None:
    _rewrite(pack_dir, lambda c: c["models"]["profiles"]["test"].pop("large"))

    with pytest.raises(PackError, match="no model for tier"):
        load_pack(pack_dir)


def test_unknown_active_profile_is_rejected(pack_dir: Path) -> None:
    _rewrite(pack_dir, lambda c: c["models"].update(active="prod"))

    with pytest.raises(PackError, match="active profile 'prod' is not defined"):
        load_pack(pack_dir)


def test_unknown_provider_and_unknown_keys_are_rejected(pack_dir: Path) -> None:
    _rewrite(pack_dir, lambda c: c["models"]["profiles"]["test"]["small"][0].update(provider="x"))
    with pytest.raises(PackError, match="invalid"):
        load_pack(pack_dir)

    _rewrite(
        pack_dir, lambda c: c["models"]["profiles"]["test"]["small"][0].update(provider="fake")
    )
    _rewrite(pack_dir, lambda c: c.update(typo_key=1))
    with pytest.raises(PackError, match="typo_key"):
        load_pack(pack_dir)


def test_missing_system_prompt_is_reported(pack_dir: Path) -> None:
    (pack_dir / "prompts" / "system.md").unlink()

    with pytest.raises(PackError, match="system prompt file not found"):
        load_pack(pack_dir)
