from loguru import logger

from src.config import ServiceConfigSchema
from src.services.stores.base import TranslationStore
from src.services.stores.local import LocalStore, UploadReport
from src.services.stores.weblate import WeblateStore
from src.services.weblate import AsyncWeblateClient


def build_store(
    config: ServiceConfigSchema, client: AsyncWeblateClient
) -> TranslationStore:
    """The store `config.store` selects, over a client for `config.weblate`."""
    if config.writes_weblate:
        return WeblateStore(client)
    return LocalStore(client, config.store_path)


async def upload_local(
    config: ServiceConfigSchema, *, language: str, slugs: set[str] | None = None
) -> UploadReport:
    """Push this machine's queued translations with `config`'s token.

    Units Weblate has changed since they were translated here are held
    back as conflicts and logged.
    """
    async with AsyncWeblateClient(config.weblate) as client:
        store = LocalStore(client, config.store_path)
        try:
            report = await store.upload(client, language=language, slugs=slugs)
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
