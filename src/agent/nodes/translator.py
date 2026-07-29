from __future__ import annotations

import asyncio
from typing import TypedDict

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.llm import TranslationAgent
from src.agent.prompts import format_translation_prompt
from src.agent.tools import lookup_glossary_or_patterns
from src.models.agent import (
    AgentInputSchema,
    NewAgentStateSchema,
    PatternSchema,
    TranslationOutputSchema,
    TranslationUnitSchema,
)
from src.models.weblate import WeblateUnitSchema


class TranslateOutputSchema(TypedDict):
    candidates: list[TranslationUnitSchema]


type _Matches = tuple[
    list[WeblateUnitSchema], list[WeblateUnitSchema], list[PatternSchema]
]


async def translator(
    state: NewAgentStateSchema,
    *,
    agent_config: AgentConfigSchema,
    agent: TranslationAgent,
) -> TranslateOutputSchema:
    matches: dict[str, _Matches] = {}

    def _matches(source: str) -> _Matches:
        """Fuzzy lookup is O(len(glossary)) per miss; do it once per source."""
        hit = matches.get(source)
        if hit is None:
            hit = matches[source] = (
                lookup_glossary_or_patterns(source, state.base_glossary),
                lookup_glossary_or_patterns(source, state.mods_glossary),
                lookup_glossary_or_patterns(source, state.patterns),
            )
        return hit

    def _build_translate_input(unit: WeblateUnitSchema) -> AgentInputSchema:
        base_matches, mods_matches, match_patterns = _matches(unit.source)
        prompt = format_translation_prompt(
            unit.source,
            unit.note,
            base_matches,
            mods_matches,
            state.context_results[unit.id],
            match_patterns,
        )
        feedback = state.quality_feedback.get(unit.id)
        if feedback:
            prompt = f"{prompt}\n\n## Previous attempt was rejected\n{feedback}"
        return {"messages": [{"role": "user", "content": prompt}]}

    def _candidate(unit: WeblateUnitSchema, translated: str) -> TranslationUnitSchema:
        base_matches, mods_matches, match_patterns = _matches(unit.source)
        return TranslationUnitSchema(
            id=unit.id,
            source=unit.source,
            translated=translated,
            key=unit.context,
            context=state.context_results[unit.id],
            category=unit.note or "unknown",
            pattern_matched=bool(match_patterns),
            glossary_base=base_matches,
            glossary_mods=mods_matches,
            tag_valid=False,
            original_unit=unit,
            patterns=match_patterns,
        )

    inputs = await asyncio.to_thread(
        lambda: [_build_translate_input(unit) for unit in state.to_translate]
    )
    responses = await agent.abatch(
        inputs, config={"max_concurrency": agent_config.max_concurrency}
    )

    candidates: list[TranslationUnitSchema] = []
    for unit, response in zip(state.to_translate, responses, strict=True):
        structured = response.get("structured_response") if response else None
        if isinstance(structured, TranslationOutputSchema):
            candidates.append(_candidate(unit, structured.result))
        else:
            logger.warning(f"No structured translation for unit {unit.id}")
            candidates.append(_candidate(unit, ""))
    logger.success(f"Translated {len(candidates)} units")
    return {"candidates": candidates}
