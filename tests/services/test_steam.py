import asyncio
from pathlib import Path

import pytest
from loguru import logger
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
    def __init__(
        self, returncode: int | None = 0, hang: bool = False, output: bytes = b""
    ) -> None:
        self.returncode = returncode
        self._output = output
        self._hang = hang
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        if self._hang:
            await asyncio.Event().wait()
        return self.returncode or 0

    async def communicate(self) -> tuple[bytes, None]:
        await self.wait()
        return self._output, None

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


async def test_failure_logs_masked_output_tail(
    tmp_path: Path, spawn: list[FakeProcess]
) -> None:
    output = (
        b"progress\r" * 50
        + b"Logging in user 'steam-user' steam-password\nFAILED (Invalid Password)\n"
    )
    spawn.append(FakeProcess(returncode=5, output=output))
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        with pytest.raises(SteamDownloadError):
            await downloader(tmp_path).download("42")
    finally:
        logger.remove(sink)
    tails = [m for m in messages if "last output:" in m]
    assert len(tails) == 2
    for message in tails:
        tail = message.split("last output:\n", 1)[1].strip().splitlines()
        assert tail[-1] == "FAILED (Invalid Password)"
        assert len(tail) == 20
        assert "steam-password" not in message


async def test_fetch_collection_items_expands_children() -> None:
    from httpx2 import AsyncClient, MockTransport, Request, Response

    from src.services.steam import (
        STEAM_COLLECTION_URL,
        fetch_collection_items,
    )

    def handler(request: Request) -> Response:
        if str(request.url) == STEAM_COLLECTION_URL:
            return Response(
                200,
                json={
                    "response": {
                        "collectiondetails": [
                            {
                                "publishedfileid": "9",
                                "result": 1,
                                "children": [
                                    {"publishedfileid": "1"},
                                    {"publishedfileid": "2"},
                                ],
                            }
                        ]
                    }
                },
            )
        return Response(
            200,
            json={
                "response": {
                    "publishedfiledetails": [
                        {
                            "publishedfileid": "1",
                            "result": 1,
                            "consumer_app_id": XCOM2_APP_ID,
                            "title": "A",
                            "file_size": "1200",
                        },
                        {"publishedfileid": "2", "result": 9},
                    ]
                }
            },
        )

    async with AsyncClient(transport=MockTransport(handler)) as client:
        items = await fetch_collection_items("9", client=client)
    assert [(i.publishedfileid, i.result, i.file_size) for i in items] == [
        ("1", 1, 1200),
        ("2", 9, 0),
    ]


async def test_fetch_collection_rejects_private_collection() -> None:
    from httpx2 import AsyncClient, MockTransport, Request, Response

    from src.core.workshop import WorkshopInputError
    from src.services.steam import fetch_collection_items

    def handler(request: Request) -> Response:
        return Response(
            200,
            json={
                "response": {
                    "collectiondetails": [{"publishedfileid": "9", "result": 9}]
                }
            },
        )

    async with AsyncClient(transport=MockTransport(handler)) as client:
        with pytest.raises(WorkshopInputError, match="not public"):
            await fetch_collection_items("9", client=client)
