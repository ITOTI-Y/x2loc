import asyncio
from pathlib import Path
from typing import Annotated

import typer

from src.config import ServiceConfigSchema
from src.models.workshop import TARGET_LANGUAGE
from src.services.stores import upload_local


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
    asyncio.run(
        upload_local(
            config, language=TARGET_LANGUAGE, slugs=set(slugs) if slugs else None
        )
    )
