from src.config import ServiceConfigSchema
from src.services.stores.base import TranslationStore
from src.services.stores.local import LocalStore
from src.services.stores.weblate import WeblateStore
from src.services.weblate import AsyncWeblateClient


def build_store(
    config: ServiceConfigSchema, client: AsyncWeblateClient
) -> TranslationStore:
    """The store `config.store` selects, over a client for `config.weblate`."""
    if config.writes_weblate:
        return WeblateStore(client)
    return LocalStore(client, config.store_path)
