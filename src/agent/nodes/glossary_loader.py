import asyncio
from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.core.glossary import group_units
from src.models.weblate import WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient


class GlossaryLoaderOutputSchema(TypedDict):
    base_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    mods_glossary: dict[str, tuple[WeblateUnitSchema, ...]]


async def glossary_loader(
    *,
    client: AsyncWeblateClient,
    agent_config: ConfigSchema,
) -> GlossaryLoaderOutputSchema:
    """Load the three glossaries fresh from Weblate and index them.

    `mods` and `custom` are merged: both are mod-scoped terminology and the
    translator consults them as one table. No cache outlives the job —
    Weblate is the authority, and a previous job may have appended custom
    terms that this job must see.
    """

    async def _load(slug: str) -> list[WeblateUnitSchema]:
        return await client.list_units(
            slug, agent_config.target_lang, q="state:translated"
        )

    base, mods, custom = await asyncio.gather(
        _load(agent_config.base_glossary_slug),
        _load(agent_config.mods_glossary_slug),
        _load(agent_config.custom_glossary_slug),
    )

    logger.success(
        "Loaded glossaries: {} base + {} mods + {} custom",
        len(base),
        len(mods),
        len(custom),
    )
    return {
        "base_glossary": group_units(base),
        "mods_glossary": group_units([*mods, *custom]),
    }
