import asyncio
from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.nodes.pattern_extractor import mine_glossary_patterns
from src.core.glossary import group_units
from src.models.agent import PatternSchema
from src.models.weblate import WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient


class GlossaryLoaderOutputSchema(TypedDict):
    base_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    mods_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    patterns: dict[str, tuple[PatternSchema, ...]]


async def glossary_loader(
    *,
    client: AsyncWeblateClient,
    agent_config: ConfigSchema,
) -> GlossaryLoaderOutputSchema:
    """Load the three glossaries fresh from Weblate and index them.

    `mods` and `custom` are merged: both are mod-scoped terminology and the
    translator consults them as one table. No cache outlives the job —
    Weblate is the authority, and a previous job may have appended custom
    terms that this job must see. Translation patterns are mined from the
    same glossaries, so they are rebuilt from Weblate on every job.
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

    base_glossary = group_units(base)
    mods_glossary = group_units([*mods, *custom])
    patterns = await asyncio.to_thread(
        mine_glossary_patterns, base_glossary, mods_glossary
    )
    logger.success(
        "Loaded glossaries: {} base + {} mods + {} custom; mined {} patterns",
        len(base),
        len(mods),
        len(custom),
        len(patterns),
    )
    return {
        "base_glossary": base_glossary,
        "mods_glossary": mods_glossary,
        "patterns": patterns,
    }
