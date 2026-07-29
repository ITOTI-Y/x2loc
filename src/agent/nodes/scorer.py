from __future__ import annotations

from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.llm import ScoringAgent, raise_if_fatal_llm_error
from src.agent.prompts import format_scoring_prompt
from src.models.agent import (
    AgentInputSchema,
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
    def build_input(unit: TranslationUnitSchema) -> AgentInputSchema:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": format_scoring_prompt(
                        source=unit.source,
                        translated=unit.translated,
                        category=unit.category,
                        base_matches=unit.glossary_base,
                        mods_matches=unit.glossary_mods,
                        context_results=unit.context,
                        patterns=unit.patterns,
                    ),
                }
            ]
        }

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

    responses = await llm.abatch(
        [build_input(candidate) for candidate in pending],
        config={"max_concurrency": agent_config.max_concurrency},
        return_exceptions=True,
    )
    for response, candidate in zip(responses, pending, strict=True):
        if isinstance(response, BaseException):
            raise_if_fatal_llm_error(response)
            logger.warning(
                f"Scoring request failed for unit {candidate.id}: {response!r}"
            )
            structured = None
        else:
            structured = response.get("structured_response") if response else None
        result = (
            structured
            if isinstance(structured, ScoreResultSchema)
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
    logger.success("Scored {} units", len(scores))
    return {"scores": scores}
