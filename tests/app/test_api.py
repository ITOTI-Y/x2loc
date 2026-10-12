from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from keyring.backend import KeyringBackend

from src.app.api import create_app
from src.app.secrets import Secrets

TOKEN = "session"
BUNDLE = Path(__file__).parents[2] / "mods" / "x2locWeaponCatNames"


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[service, username] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


@pytest.fixture
def keyring() -> MemoryKeyring:
    return MemoryKeyring()


@pytest.fixture
def client(tmp_path: Path, keyring: MemoryKeyring) -> Iterator[TestClient]:
    app = create_app(
        home=tmp_path / "home",
        session_token=TOKEN,
        secrets=Secrets(keyring),
        bundle=BUNDLE,
    )
    with TestClient(
        app, base_url="http://127.0.0.1:8000", headers={"X-Session-Token": TOKEN}
    ) as test_client:
        yield test_client


def local_settings(client: TestClient, tmp_path: Path) -> Path:
    mods_dir = tmp_path / "Mods"
    workshop_dir = tmp_path / "workshop"
    mods_dir.mkdir()
    workshop_dir.mkdir()
    settings = client.get("/api/settings").json()["settings"]
    settings["store"] = "local"
    settings["paths"] = {"workshop_dir": str(workshop_dir), "mods_dir": str(mods_dir)}
    assert client.put("/api/settings", json=settings).status_code == 200
    return mods_dir


def test_requests_without_the_session_token_are_refused(client: TestClient) -> None:
    response = client.get("/api/settings", headers={"X-Session-Token": "wrong"})

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_the_token_may_come_as_a_query_for_event_streams(client: TestClient) -> None:
    response = client.get(
        f"/api/settings?token={TOKEN}", headers={"X-Session-Token": ""}
    )

    assert response.status_code == 200


def test_foreign_hosts_are_refused(client: TestClient) -> None:
    response = client.get("/api/settings", headers={"Host": "evil.example:8000"})

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden_host"


def test_settings_persist_and_secrets_report_only_presence(
    client: TestClient, keyring: MemoryKeyring
) -> None:
    settings = client.get("/api/settings").json()["settings"]
    settings["model"]["translation_model"] = "some/model"
    client.put("/api/settings", json=settings)
    put = client.put("/api/secrets/llm_api_key", json={"value": "sk-secret"})

    view = client.get("/api/settings").json()

    assert put.status_code == 204
    assert view["settings"]["model"]["translation_model"] == "some/model"
    assert view["secrets"] == {"llm_api_key": True, "weblate_token": False}
    assert "sk-secret" not in client.get("/api/settings").text
    assert keyring.values["x2loc", "llm_api_key"] == "sk-secret"

    client.delete("/api/secrets/llm_api_key")
    assert client.get("/api/settings").json()["secrets"]["llm_api_key"] is False


def test_a_priority_that_is_not_a_permutation_is_rejected(client: TestClient) -> None:
    settings = client.get("/api/settings").json()["settings"]
    settings["priority"] = ["manual", "manual", "machine", "needs_editing", "shipped"]

    response = client.put("/api/settings", json=settings)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_mods_need_paths(client: TestClient) -> None:
    response = client.get("/api/mods")

    assert response.status_code == 409
    assert response.json()["code"] == "paths_not_set"


def test_a_run_needs_a_translation_model(client: TestClient) -> None:
    response = client.post("/api/runs", json={"workshop_ids": ["1"]})

    assert response.json()["code"] == "model_not_set"


def test_a_run_rejects_ids_that_are_not_workshop_ids(client: TestClient) -> None:
    response = client.post("/api/runs", json={"workshop_ids": ["../x"]})

    assert response.status_code == 422


def test_overlays_are_listed_and_uninstalled(
    client: TestClient, tmp_path: Path
) -> None:
    mods_dir = local_settings(client, tmp_path)
    overlay = mods_dir / "x2loc_123_zh"
    overlay.mkdir()
    (overlay / "x2loc_123_zh.XComMod").write_text(
        "[mod]\nTitle=Some Mod [zh_Hans]\n", encoding="utf-8"
    )
    (mods_dir / "OtherMod").mkdir()

    listed = client.get("/api/overlays").json()
    deleted = client.delete("/api/overlays/123")

    assert listed == {
        "overlays": [{"workshop_id": "123", "title": "Some Mod [zh_Hans]"}],
        "weapon_names_installed": False,
    }
    assert deleted.status_code == 204
    assert not overlay.exists()
    assert client.delete("/api/overlays/123").json()["code"] == "overlay_not_found"


def test_the_weapon_category_mod_installs_its_prebuilt_script(
    client: TestClient, tmp_path: Path
) -> None:
    mods_dir = local_settings(client, tmp_path)

    assert client.put("/api/weapon-names").status_code == 204

    installed = mods_dir / "x2locWeaponCatNames"
    assert (installed / "Script" / "x2locWeaponCatNames.u").is_file()
    assert (installed / "Config" / "XComGame.ini").is_file()
    assert not (installed / "Src").exists()
    assert client.get("/api/overlays").json()["weapon_names_installed"] is True

    client.delete("/api/weapon-names")
    assert not installed.exists()


def test_a_local_review_edit_waits_for_upload(
    client: TestClient, tmp_path: Path
) -> None:
    local_settings(client, tmp_path)
    edit = {
        "slug": "mod-a",
        "context": "X.Y",
        "source": "Abort ",
        "target": "放弃 ",
        "upstream": "撤销",
    }

    response = client.put("/api/review", json=edit)
    queued = client.get("/api/uploads").json()

    expected = {
        "slug": "mod-a",
        "context": "X.Y",
        "origin": "manual",
        "source": "Abort ",
        "target": "放弃 ",
        "upload": "pending",
        "upstream_at_read": "撤销",
    }
    assert response.status_code == 204
    assert [{key: row[key] for key in expected} for row in queued] == [expected]


def test_uploading_needs_a_token(client: TestClient, tmp_path: Path) -> None:
    local_settings(client, tmp_path)

    response = client.post("/api/uploads", json={})

    assert response.status_code == 409
    assert response.json()["code"] == "token_not_set"


def test_a_run_refuses_mods_that_are_not_subscribed(
    client: TestClient, tmp_path: Path
) -> None:
    local_settings(client, tmp_path)
    settings = client.get("/api/settings").json()["settings"]
    settings["model"]["translation_model"] = "some/model"
    client.put("/api/settings", json=settings)

    response = client.post("/api/runs", json={"workshop_ids": ["404"]})

    assert response.status_code == 409
    assert response.json()["code"] == "mod_not_found"
