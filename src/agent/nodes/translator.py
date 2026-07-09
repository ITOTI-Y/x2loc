from __future__ import annotations

import asyncio
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.llm import TranslationOutputSchema
from src.agent.prompts import format_translation_prompt, translation_system_blocks
from src.agent.state import (
    AgentState,
    ContextResult,
    SessionPattern,
    TranslationCandidate,
    TranslationUnit,
)
from src.agent.tools import lookup_glossary, match_session_patterns


async def translator(
    state: AgentState, *, agent_config: AgentConfigSchema, llm: Runnable
) -> dict:
    system_blocks = translation_system_blocks(agent_config.target_lang)
    candidates: list[TranslationCandidate] = []
    context_map = {c["unit_id"]: c for c in state["context_results"]}
    to_translate: list[tuple[TranslationUnit, ContextResult]] = []

    compiled_patterns = _compile_patterns(state["session_patterns"])

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
            compiled_patterns,
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
    ) -> TranslationCandidate | None:
        base_matches = lookup_glossary(unit["source"], state["base_glossary"])
        mods_matches = lookup_glossary(unit["source"], state["mods_glossary"])
        match_patterns = match_session_patterns(
            unit["source"], state["session_patterns"]
        )
        prompt = format_translation_prompt(
            unit["source"],
            unit["category"],
            base_matches,
            mods_matches,
            ctx,
            match_patterns,
        )
        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_blocks),
                    HumanMessage(content=prompt),
                ]
            )
            if not isinstance(response, TranslationOutputSchema):
                raise TypeError(
                    f"Expected TranslationOutputSchema, got {type(response).__name__}"
                )
        except Exception as e:
            logger.error(f"Translation failed for {unit['source']}: {e}")
            return None
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
    failed_ids = [
        u["id"]
        for (u, _), r in zip(to_translate, llm_results, strict=False)
        if r is None
    ]
    candidates.extend(r for r in llm_results if r is not None)
    if failed_ids:
        return {
            "candidates": candidates,
            "skip_ids": list(state["skip_ids"]) + failed_ids,
        }
    return {"candidates": candidates}


def _compile_patterns(
    patterns: list[SessionPattern],
) -> list[tuple[re.Pattern[str], SessionPattern]]:
    """Compile each session pattern's src regex once per translator call."""
    return [
        (
            re.compile(re.escape(p["src_pattern"]).replace(r"\{X\}", "(.+)")),
            p,
        )
        for p in patterns
    ]


def _try_pattern_match(
    source: str,
    compiled_patterns: list[tuple[re.Pattern[str], SessionPattern]],
    base_glossary: dict[str, dict],
    mods_glossary: dict[str, dict],
) -> str | None:
    for regex, pattern in compiled_patterns:
        m = regex.fullmatch(source)
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
