import asyncio
import json
from pathlib import Path
from typing import Literal, TypedDict

from loguru import logger

from src._share import TEMP_DIR
from src.agent.config import AgentConfigSchema
from src.models.agent import NewAgentStateSchema
from src.models.weblate import WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient


class GlossaryLoaderOutputSchema(TypedDict):
    base_glossary: dict[str, WeblateUnitSchema]
    mods_glossary: dict[str, WeblateUnitSchema]


async def glossary_loader(
    state: NewAgentStateSchema,
    *,
    client: AsyncWeblateClient,
    agent_config: AgentConfigSchema,
) -> GlossaryLoaderOutputSchema:

    base, mods = await asyncio.gather(
        _load_data("base", agent_config.target_lang, agent_config, client),
        _load_data("mods", agent_config.target_lang, agent_config, client),
    )

    logger.info(f"Loaded glossaries: {len(base)} base + {len(mods)} mods")
    return {
        "base_glossary": base,
        "mods_glossary": mods,
    }


def _cache_path(mode: Literal["base", "mods"], lang: str) -> Path:
    return TEMP_DIR / f"{mode}_{lang}.json"


async def _load_data(
    mode: Literal["base", "mods"],
    lang: str,
    agent_config: AgentConfigSchema,
    client: AsyncWeblateClient,
) -> dict[str, WeblateUnitSchema]:
    path = _cache_path(mode, lang)
    if path.exists():
        result: dict[str, WeblateUnitSchema] = {}
        for item in json.loads(path.read_text("utf-8")):
            unit = WeblateUnitSchema.model_validate(item)
            result[unit.source] = unit
        return result
    if mode == "base":
        slug = agent_config.base_glossary_slug
    else:
        slug = agent_config.component_slug
    data = await client.list_units(slug, lang, q="state:translated")
    _save_data(mode, lang, data)
    return {unit.source: unit for unit in data}


def _save_data(
    mode: Literal["base", "mods"], lang: str, data: list[WeblateUnitSchema]
) -> None:
    path = _cache_path(mode, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = [unit.model_dump() for unit in data]
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
