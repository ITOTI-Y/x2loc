from __future__ import annotations

import asyncio
import re

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.llm import TranslationOutputSchema, build_translator_llm
from src.agent.prompts import TRANSLATION_SYSTEM, format_translation_prompt
from src.agent.state import (
    AgentState,
    ContextResult,
    SessionPattern,
    TranslationCandidate,
    TranslationUnit,
)
from src.agent.tools import lookup_glossary


async def translator(state: AgentState, *, agent_config: AgentConfigSchema) -> dict:
    translator_llm = build_translator_llm(agent_config)
    candidates: list[TranslationCandidate] = []
    context_map = {c["unit_id"]: c for c in state["context_results"]}
    to_translate: list[tuple[TranslationUnit, ContextResult]] = []

    for unit in state["current_units"]:
        ctx = context_map.get(
            unit["id"],
            {
                "unit_id": unit["id"],
                "mod_component": None,
                "translated_percent": None,
                "nearby": [],
            },
        )
        pattern_result = _try_pattern_match(
            unit["source"],
            state["session_patterns"],
            state["base_glossary"],
            state["mods_glossary"],
        )
        if pattern_result:
            candidates.append(
                TranslationCandidate(
                    unit_id=unit["id"],
                    source=unit["source"],
                    context=unit["context"],
                    category=unit["category"],
                    translation=pattern_result,
                    pattern_matched=True,
                    glossary_base=[],
                    glossary_mods=[],
                    context_result=ctx,
                    tag_valid=True,
                )
            )
            logger.info(f"[PATTERN] {unit['source']} → {pattern_result}")
        else:
            to_translate.append((unit, ctx))

    if not to_translate:
        return {"candidates": candidates}

    async def _translate_one(
        unit: TranslationUnit, ctx: ContextResult
    ) -> TranslationCandidate:
        base_matches = lookup_glossary(unit["source"], state["base_glossary"])
        mods_matches = lookup_glossary(unit["source"], state["mods_glossary"])
        match_patterns = []
        for single_word in unit["source"].split():
            match_patterns.extend(i for i in state["session_patterns"] if single_word.lower() in i["src_pattern"].lower())
        prompt = format_translation_prompt(
            unit["source"],
            unit["category"],
            base_matches,
            mods_matches,
            ctx,
            match_patterns,
        )
        response = await translator_llm.ainvoke(
            [
                SystemMessage(
                    content=TRANSLATION_SYSTEM.format(
                        target_lang=agent_config.target_lang
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        if not isinstance(response, TranslationOutputSchema):
            raise TypeError(
                f"Expected TranslationOutputSchema, got {type(response).__name__}"
            )
        result = response.result
        logger.info(f"[TRANSLATE] {unit['source']} → {result}")
        return TranslationCandidate(
            unit_id=unit["id"],
            source=unit["source"],
            context=unit["context"],
            category=unit["category"],
            translation=result,
            pattern_matched=False,
            glossary_base=base_matches,
            glossary_mods=mods_matches,
            context_result=ctx,
            tag_valid=False,
        )

    llm_results = await asyncio.gather(*[_translate_one(u, c) for u, c in to_translate])
    candidates.extend(llm_results)
    return {"candidates": candidates}


def _try_pattern_match(
    source: str,
    patterns: list[SessionPattern],
    base_glossary: dict[str, dict],
    mods_glossary: dict[str, dict],
) -> str | None:
    for pattern in patterns:
        regex = re.escape(pattern["src_pattern"]).replace(r"\{X\}", "(.+)")
        m = re.fullmatch(regex, source)
        if not m:
            continue
        variable = m.group(1)
        if variable in base_glossary:
            translated = base_glossary[variable]["target"]
        elif variable in mods_glossary:
            translated = mods_glossary[variable]["target"]
        else:
            continue
        return pattern["tgt_pattern"].replace("{X}", translated)
    return None
