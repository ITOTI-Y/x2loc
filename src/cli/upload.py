import asyncio
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from src.config import ServiceConfigSchema
from src.models.workshop import TARGET_LANGUAGE
from src.services.stores import LocalStore, UploadReport
from src.services.weblate import AsyncWeblateClient


def upload(
    slugs: Annotated[
        list[str] | None,
        typer.Argument(help="Component slugs to upload; every queued one if omitted."),
    ] = None,
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Service TOML.")
    ] = Path("configs/weblate.local.toml"),
) -> None:
    """Push the translations kept on this machine to Weblate.

    Units Weblate has changed since they were translated here are held
    back as conflicts and listed.
    """
    config = ServiceConfigSchema.from_toml(config_path)
    if not config.weblate.token.get_secret_value():
        raise typer.BadParameter("uploading needs a token in the [weblate] table")
    asyncio.run(_upload(config, set(slugs) if slugs else None))


async def _upload(config: ServiceConfigSchema, slugs: set[str] | None) -> UploadReport:
    async with AsyncWeblateClient(config.weblate) as client:
        store = LocalStore(client, config.store_path)
        try:
            report = await store.upload(client, language=TARGET_LANGUAGE, slugs=slugs)
            for row in store.queue(uploads=("conflict",)):
                logger.warning(
                    "Conflict in {} [{}]: here {!r}, Weblate now {!r}",
                    row.slug,
                    row.context,
                    row.target,
                    row.upstream_now,
                )
        finally:
            await store.aclose()
    return report
