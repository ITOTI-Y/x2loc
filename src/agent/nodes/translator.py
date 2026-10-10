from __future__ import annotations

import asyncio
from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.llm import TranslationAgent
from src.agent.nodes._batched import invoke_batched
from src.agent.prompts import format_translation_prompt
from src.agent.tools import lookup_glossary, match_patterns
from src.core.placeholders import repair_markup
from src.models.agent import (
    NewAgentStateSchema,
    PatternSchema,
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
    agent_config: ConfigSchema,
    llm_slots: asyncio.Semaphore,
    agent: TranslationAgent,
) -> TranslateOutputSchema:
    matches: dict[str, _Matches] = {}

    def _matches(source: str) -> _Matches:
        """Lookups scan every glossary key and template; do them once per source."""
        hit = matches.get(source)
        if hit is None:
            hit = matches[source] = (
                lookup_glossary(source, state.base_glossary),
                lookup_glossary(source, state.mods_glossary),
                match_patterns(source, state.patterns),
            )
        return hit

    def _build_prompt(unit: WeblateUnitSchema) -> tuple[int, str]:
        base_matches, mods_matches, match_patterns = _matches(unit.source)
        prompt = format_translation_prompt(
            repair_markup(unit.source),
            unit.note,
            base_matches,
            mods_matches,
            state.context_results[unit.id],
            match_patterns,
        )
        feedback = state.quality_feedback.get(unit.id)
        if feedback:
            prompt = f"{prompt}\n\nPrevious attempt was rejected:\n{feedback}"
        return unit.id, prompt

    def _candidate(unit: WeblateUnitSchema, translated: str) -> TranslationUnitSchema:
        base_matches, mods_matches, match_patterns = _matches(unit.source)
        return TranslationUnitSchema(
            id=unit.id,
            # The validator and scorer judge against the repaired markup the
            # translator was shown.
            source=repair_markup(unit.source),
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

    prompts = await asyncio.to_thread(
        lambda: [_build_prompt(unit) for unit in state.to_translate]
    )
    results = await invoke_batched(
        agent,
        prompts,
        units_per_request=agent_config.units_per_request,
        slots=llm_slots,
        label="Translation",
    )
    candidates = [
        _candidate(unit, results[unit.id].result if unit.id in results else "")
        for unit in state.to_translate
    ]
    logger.success(f"Translated {len(candidates)} units")
    return {"candidates": candidates}
