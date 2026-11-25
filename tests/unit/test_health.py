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


def test_the_demo_page_is_served_unless_turned_off(pack_dir: Path) -> None:
    from fastapi.testclient import TestClient

    from core.api.app import create_app
    from core.config import Settings

    with_page = TestClient(create_app(Settings(pack=pack_dir)))
    without = TestClient(create_app(Settings(pack=pack_dir, demo_page=False)))

    page = with_page.get("/demo/")
    assert page.status_code == 200 and "/v1/chat" in page.text
    assert without.get("/demo/").status_code == 404
