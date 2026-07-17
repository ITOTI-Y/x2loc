from __future__ import annotations

from typing import TypedDict

from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agent.config import AgentConfigSchema
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
    agent_config: AgentConfigSchema,
    llm: CompiledStateGraph,
) -> ScorerOutputSchema:
    def _build_score_input(unit: TranslationUnitSchema) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": format_scoring_prompt(
                        source=unit.source,
                        tanslated=unit.translated,
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
    to_score: list[TranslationUnitSchema] = []
    for c in state.candidates:
        if not c.tag_valid:
            scores.append(
                c.model_copy(
                    update={
                        "score_result": ScoreResultSchema(
                            score=0,
                            deductions=[],
                            suggested_translation=None,
                            notes="tag-not-valid",
                        )
                    }
                )
            )
        else:
            to_score.append(c)

    if to_score:
        responses = await llm.abatch([_build_score_input(c) for c in to_score])
        for response, unit in zip(responses, to_score, strict=True):
            if response is not None:
                structured = response.get("structured_response")
                if isinstance(structured, ScoreResultSchema):
                    scores.append(
                        unit.model_copy(
                            update={
                                "score_result": structured,
                                "suggested_translation": structured.suggested_translation,
                            }
                        )
                    )
                else:
                    scores.append(
                        unit.model_copy(
                            update={
                                "score_result": ScoreResultSchema(
                                    score=0,
                                    deductions=[],
                                    suggested_translation=None,
                                    notes="no-score-result",
                                ),
                                "suggested_translation": None,
                            }
                        )
                    )
    logger.success(f"Scored {len(scores)} units")
    return {"scores": scores}
