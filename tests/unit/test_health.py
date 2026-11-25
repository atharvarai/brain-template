from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.config import PackError, Settings


def test_healthz_reports_pack_and_profile(pack_dir: Path) -> None:
    client = TestClient(create_app(Settings(pack=pack_dir)))

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["brain"] == "Acme"
    assert body["model_profile"] == "test"


def test_env_profile_overrides_the_active_one(pack_dir: Path) -> None:
    client = TestClient(create_app(Settings(pack=pack_dir, model_profile="alt")))

    assert client.get("/healthz").json()["model_profile"] == "alt"


def test_unknown_profile_override_fails_at_startup(pack_dir: Path) -> None:
    with pytest.raises(PackError, match="model profile 'nope' is not defined"):
        create_app(Settings(pack=pack_dir, model_profile="nope"))


def test_the_demo_page_is_served_only_when_its_folder_exists(
    pack_dir: Path, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from core.api.app import create_app
    from core.config import Settings

    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>demo</title>")

    with_page = TestClient(create_app(Settings(pack=pack_dir, web_dir=web)))
    without = TestClient(create_app(Settings(pack=pack_dir, web_dir=None)))

    assert with_page.get("/demo/").status_code == 200
    assert "demo" in with_page.get("/demo/").text
    assert without.get("/demo/").status_code == 404
