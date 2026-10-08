from __future__ import annotations

from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.llm import ScoringAgent
from src.agent.nodes._batched import invoke_batched
from src.agent.prompts import format_scoring_prompt
from src.models.agent import (
    NewAgentStateSchema,
    ScoreResultSchema,
    TranslationUnitSchema,
)


class ScorerOutputSchema(TypedDict):
    scores: list[TranslationUnitSchema]


async def scorer(
    state: NewAgentStateSchema,
    *,
    agent_config: ConfigSchema,
    llm: ScoringAgent,
) -> ScorerOutputSchema:
    def build_prompt(unit: TranslationUnitSchema) -> tuple[int, str]:
        return unit.id, format_scoring_prompt(
            source=unit.source,
            translated=unit.translated,
            category=unit.category,
            base_matches=unit.glossary_base,
            mods_matches=unit.glossary_mods,
            context_results=unit.context,
            patterns=unit.patterns,
        )

    scores: list[TranslationUnitSchema] = []
    pending: list[TranslationUnitSchema] = []
    for candidate in state.candidates:
        if candidate.tag_valid:
            pending.append(candidate)
        else:
            scores.append(
                candidate.model_copy(
                    update={
                        "score_result": ScoreResultSchema(
                            score=0,
                            deductions=[],
                            suggested_translation="",
                            notes="tag-not-valid",
                        )
                    }
                )
            )

    results = await invoke_batched(
        llm,
        [build_prompt(candidate) for candidate in pending],
        units_per_request=agent_config.units_per_request,
        max_concurrency=agent_config.max_concurrency,
        label="Scoring",
    )
    for candidate in pending:
        item = results.get(candidate.id)
        result = (
            ScoreResultSchema.model_validate(item.model_dump(exclude={"id"}))
            if item is not None
            else ScoreResultSchema(
                score=0,
                deductions=[],
                suggested_translation="",
                notes="no-score-result",
            )
        )
        scores.append(
            candidate.model_copy(
                update={
                    "score_result": result,
                    "suggested_translation": result.suggested_translation,
                }
            )
        )
    logger.success(
        "Scored {} units (attempt {}): {}",
        len(scores),
        state.attempts + 1,
        [unit.score_result.score for unit in scores if unit.score_result],
    )
    return {"scores": scores}
