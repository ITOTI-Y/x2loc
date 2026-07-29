import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.models.workshop import XCOM2_APP_ID, WorkshopLimitsSchema
from src.services.steam import SteamDownloader, SteamDownloadError
from tests.conftest import _write_loc_file


def limits(**overrides: float) -> WorkshopLimitsSchema:
    values: dict[str, float] = {
        "download_timeout_seconds": 5.0,
        "terminate_grace_seconds": 0.1,
        "max_total_bytes": 10_000_000,
        "max_file_count": 100,
        "max_loc_file_bytes": 1_000_000,
    }
    values.update(overrides)
    return WorkshopLimitsSchema.model_validate(values)


def downloader(tmp_path: Path, **overrides: float) -> SteamDownloader:
    return SteamDownloader(
        executable=Path("/usr/bin/steamcmd"),
        steam_root=tmp_path,
        username="steam-user",
        password=SecretStr("steam-password"),
        limits=limits(**overrides),
    )


class FakeProcess:
    def __init__(self, returncode: int | None = 0, hang: bool = False) -> None:
        self.returncode = returncode
        self._hang = hang
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        if self._hang:
            await asyncio.Event().wait()
        return self.returncode or 0

    def terminate(self) -> None:
        self.terminated = True
        self._hang = False
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.fixture
def spawn(monkeypatch: pytest.MonkeyPatch) -> list[FakeProcess]:
    spawned: list[FakeProcess] = []
    argv: list[tuple[str, ...]] = []

    async def fake_exec(*args: str, **_kwargs: object) -> FakeProcess:
        argv.append(args)
        return spawned[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return spawned


def content_dir(tmp_path: Path, workshop_id: str) -> Path:
    return (
        tmp_path
        / "steamapps"
        / "workshop"
        / "content"
        / str(XCOM2_APP_ID)
        / workshop_id
    )


def _write_xcommod(mod_root: Path, title: str = "Test Mod") -> None:
    """Write the `.XComMod` manifest every real Workshop item carries."""
    mod_root.mkdir(parents=True, exist_ok=True)
    (mod_root / "TestMod.XComMod").write_text(
        f"[mod]\npublishedFileId=0\nTitle={title}\n", encoding="utf-8"
    )


async def test_non_zero_exit_raises(tmp_path: Path, spawn: list[FakeProcess]) -> None:
    spawn.append(FakeProcess(returncode=1))
    with pytest.raises(SteamDownloadError, match="non-zero"):
        await downloader(tmp_path).download("42")


async def test_missing_content_dir_raises(
    tmp_path: Path, spawn: list[FakeProcess]
) -> None:
    spawn.append(FakeProcess(returncode=0))
    with pytest.raises(SteamDownloadError, match="no content directory"):
        await downloader(tmp_path).download("42")


async def test_missing_manifest_raises(
    tmp_path: Path, spawn: list[FakeProcess]
) -> None:
    spawn.append(FakeProcess(returncode=0))
    root = content_dir(tmp_path, "42")
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    with pytest.raises(SteamDownloadError, match="not a resolvable mod"):
        await downloader(tmp_path).download("42")


async def test_timeout_terminates_process(
    tmp_path: Path, spawn: list[FakeProcess]
) -> None:
    process = FakeProcess(returncode=None, hang=True)
    spawn.append(process)
    with pytest.raises(SteamDownloadError, match="timed out"):
        await downloader(tmp_path, download_timeout_seconds=0.05).download("42")
    assert process.terminated


async def test_cancel_terminates_process(
    tmp_path: Path, spawn: list[FakeProcess]
) -> None:
    process = FakeProcess(returncode=None, hang=True)
    spawn.append(process)
    task = asyncio.create_task(downloader(tmp_path).download("42"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated


async def test_successful_download_returns_scanned_tree(
    tmp_path: Path, spawn: list[FakeProcess]
) -> None:
    spawn.append(FakeProcess(returncode=0))
    root = content_dir(tmp_path, "42")
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    _write_xcommod(root)
    item = await downloader(tmp_path).download("42")
    assert item.workshop_id == "42"
    assert item.mod_info.steam_id == "42"
    assert item.mod_root == root.resolve()
    assert any(path.name == "Foo.int" for path in item.files)


async def test_password_never_reaches_logs(
    tmp_path: Path, spawn: list[FakeProcess], caplog: pytest.LogCaptureFixture
) -> None:
    spawn.append(FakeProcess(returncode=0))
    root = content_dir(tmp_path, "42")
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    _write_xcommod(root)
    await downloader(tmp_path).download("42")
    assert "steam-password" not in caplog.text
