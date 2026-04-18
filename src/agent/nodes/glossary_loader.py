from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.state import AgentState
from src.services.weblate import WeblateClient


def glossary_loader(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    base: dict[str, dict] = {}
    base_datas = _load_data("base", agent_config.target_lang, agent_config, client)
    for unit in base_datas:
        base[unit["source"][0]] = {
            "target": unit["target"][0],
            "context": unit["context"],
        }

    mods: dict[str, dict] = {}
    mods_datas = _load_data("mods", agent_config.target_lang, agent_config, client)
    for unit in mods_datas:
        mods[unit["source"][0]] = {
            "target": unit["target"][0],
            "context": unit["context"],
        }

    logger.info(f"Loaded glossaries: {len(base)} base + {len(mods)} mods")
    return {
        "base_glossary": base,
        "mods_glossary": mods,
        "stats": {"auto": 0, "approved": 0, "modified": 0, "skipped": 0},
        "skip_ids": [],
        "session_patterns": [],
        "current_page": 1,
        "should_continue": True,
    }


def _cache_path(mode: Literal["base", "mods"], lang: str) -> Path:
    return Path(f"temp/{mode}_{lang}.json")


def _load_data(
    mode: Literal["base", "mods"],
    lang: str,
    agent_config: AgentConfigSchema,
    client: WeblateClient,
) -> list[dict]:
    path = _cache_path(mode, lang)
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    if mode == "base":
        slug = agent_config.base_glossary_slug
    else:
        slug = agent_config.component_slug
    data = list(client.list_units(slug, lang, q="state:translated"))
    _save_data(mode, lang, data)
    return data


def _save_data(mode: Literal["base", "mods"], lang: str, data: list[dict]) -> None:
    path = _cache_path(mode, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
