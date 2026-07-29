import asyncio
from pathlib import Path

from loguru import logger
from pydantic import SecretStr

from src.core.mod_resolver import ModResolveError, resolve_mod
from src.core.workshop import scan_mod_tree
from src.models.workshop import XCOM2_APP_ID, WorkshopItemSchema, WorkshopLimitsSchema


class SteamDownloadError(RuntimeError):
    """SteamCMD did not produce a usable Workshop item."""


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
