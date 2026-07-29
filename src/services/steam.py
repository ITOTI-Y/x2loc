import asyncio
from pathlib import Path
from typing import Final

from httpx2 import AsyncClient
from loguru import logger
from pydantic import SecretStr

from src.core.mod_resolver import ModResolveError, resolve_mod
from src.core.workshop import WorkshopInputError, scan_mod_tree
from src.models.workshop import (
    XCOM2_APP_ID,
    WorkshopDetailsEnvelopeSchema,
    WorkshopItemSchema,
    WorkshopLimitsSchema,
    WorkshopMetadataSchema,
)


class SteamDownloadError(RuntimeError):
    """SteamCMD did not produce a usable Workshop item."""


STEAM_DETAILS_URL: Final[str] = (
    "https://api.steampowered.com/ISteamRemoteStorage/GetPublishedFileDetails/v1/"
)
STEAM_RESULT_OK: Final[int] = 1


async def fetch_workshop_metadata(
    workshop_id: str, *, client: AsyncClient
) -> WorkshopMetadataSchema:
    """Anonymous metadata preflight against the official Steam Web API.

    XCOM 2 items carry no `file_url` (verified 2026-07-29: `file_size=0`,
    content exists only as an `hcontent_file` depot handle), so this call
    cannot replace the SteamCMD download. It rejects nonexistent or
    foreign-app items before SteamCMD starts and supplies `title` and
    `time_updated` for job display.
    """
    response = await client.post(
        STEAM_DETAILS_URL,
        data={"itemcount": "1", "publishedfileids[0]": workshop_id},
    )
    if response.status_code != 200:
        raise SteamDownloadError(f"Steam Web API returned {response.status_code}")
    try:
        payload = WorkshopDetailsEnvelopeSchema.model_validate(response.json())
    except ValueError as exc:
        raise SteamDownloadError("Steam Web API returned an invalid response") from exc
    if not payload.response.publishedfiledetails:
        raise WorkshopInputError(f"Workshop item {workshop_id} does not exist")
    metadata = payload.response.publishedfiledetails[0]
    if metadata.result != STEAM_RESULT_OK:
        raise WorkshopInputError(f"Workshop item {workshop_id} does not exist")
    if metadata.consumer_app_id != XCOM2_APP_ID:
        raise WorkshopInputError(f"Workshop item {workshop_id} is not an XCOM 2 mod")
    return metadata


class SteamDownloader:
    def __init__(
        self,
        *,
        executable: Path,
        steam_root: Path,
        username: str,
        password: SecretStr,
        limits: WorkshopLimitsSchema,
    ):
        self._executable = executable
        self._steam_root = steam_root
        self._username = username
        self._password = password
        self._limits = limits

    async def download(self, workshop_id: str) -> WorkshopItemSchema:
        await self._run_steamcmd(workshop_id)
        mod_root = (
            self._steam_root
            / "steamapps"
            / "workshop"
            / "content"
            / f"{XCOM2_APP_ID}"
            / f"{workshop_id}"
        )
        if not mod_root.is_dir():
            raise SteamDownloadError("SteamCMD produced no content directory")
        files = await asyncio.to_thread(scan_mod_tree, mod_root, self._limits)
        try:
            mod_info = await asyncio.to_thread(
                resolve_mod, mod_root, mod_root, workshop_id
            )
        except ModResolveError as exc:
            raise SteamDownloadError("Workshop item is not a resolvable mod") from exc
        logger.info("Downloaded Workshop item {} ({} files)", workshop_id, len(files))
        return WorkshopItemSchema(
            workshop_id=workshop_id,
            mod_root=mod_root.resolve(strict=True),
            mod_info=mod_info,
            files=files,
        )

    async def _run_steamcmd(self, workshop_id: str) -> None:
        process = await asyncio.create_subprocess_exec(
            str(self._executable),
            "+force_install_dir",
            str(self._steam_root),
            "+login",
            self._username,
            self._password.get_secret_value(),
            "+workshop_download_item",
            str(XCOM2_APP_ID),
            workshop_id,
            "+quit",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self._steam_root,
        )
        try:
            await asyncio.wait_for(
                process.wait(), timeout=self._limits.download_timeout_seconds
            )
        except TimeoutError as exc:
            raise SteamDownloadError("SteamCMD timed out") from exc
        finally:
            await self._terminate(process)
        if process.returncode != 0:
            raise SteamDownloadError("SteamCMD exited with a non-zero status")

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=self._limits.terminate_grace_seconds,
            )
        except TimeoutError:
            process.kill()
            await asyncio.shield(process.wait())
