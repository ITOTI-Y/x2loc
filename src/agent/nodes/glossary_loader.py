import asyncio
from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.nodes.pattern_extractor import mine_glossary_patterns
from src.core.glossary import group_units
from src.models.agent import PatternSchema
from src.models.weblate import WeblateUnitSchema
from src.services.glossary import GlossarySnapshots


class GlossaryLoaderOutputSchema(TypedDict):
    base_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    mods_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    patterns: dict[str, tuple[PatternSchema, ...]]


async def load_glossaries(
    snapshots: GlossarySnapshots, config: ConfigSchema
) -> GlossaryLoaderOutputSchema:
    """Index the three glossaries and mine translation patterns from them.

    `mods` and `custom` are merged: both are mod-scoped terminology and the
    translator consults them as one table.
    """
    base, mods, custom = await asyncio.gather(
        *(
            snapshots.units(slug, config.target_lang)
            for slug in (
                config.base_glossary_slug,
                config.mods_glossary_slug,
                config.custom_glossary_slug,
            )
        )
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
