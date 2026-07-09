from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from loguru import logger

from src.agent._share import GLOSSARY_CACHE_DIR
from src.agent.config import AgentConfigSchema
from src.agent.state import AgentState
from src.agent.tools import make_glossary_entry
from src.services.weblate import WeblateClient


def glossary_loader(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    def _index(units: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for unit in units:
            src = unit.get("source") or []
            tgt = unit.get("target") or []
            if not (isinstance(src, list) and src and isinstance(tgt, list) and tgt):
                logger.warning(
                    f"Skipping glossary unit with empty source/target: id={unit.get('id')}"
                )
                continue
            out[src[0]] = make_glossary_entry(src[0], tgt[0], unit.get("context", ""))
        return out

    base = _index(_load_data("base", agent_config.target_lang, agent_config, client))
    mods = _index(_load_data("mods", agent_config.target_lang, agent_config, client))

    logger.info(f"Loaded glossaries: {len(base)} base + {len(mods)} mods")
    return {
        "base_glossary": base,
        "mods_glossary": mods,
        "stats": {"auto": 0, "approved": 0, "modified": 0, "skipped": 0},
        "skip_ids": [],
        "current_page": 1,
        "should_continue": True,
    }


def _cache_path(mode: Literal["base", "mods"], lang: str) -> Path:
    return GLOSSARY_CACHE_DIR / f"{mode}_{lang}.json"


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
