from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.llm import ScoreOutputSchema
from src.agent.prompts import format_scoring_prompt, scoring_system_blocks
from src.agent.state import AgentState, Deduction, ScoreResult, TranslationCandidate
from src.agent.tools import lookup_glossary_or_patterns


async def scorer(
    state: AgentState, *, agent_config: AgentConfigSchema, llm: Runnable
) -> dict:
    system_blocks = scoring_system_blocks(agent_config.target_lang)
    instant: list[ScoreResult] = []
    to_score: list[TranslationCandidate] = []

    for c in state["candidates"]:
        if c["pattern_matched"]:
            instant.append(
                {
                    "unit_id": c["unit_id"],
                    "score": 100,
                    "deductions": [],
                    "suggested_translation": None,
                    "notes": "pattern-matched",
                }
            )
        elif not c["tag_valid"]:
            instant.append(
                ScoreResult(
                    unit_id=c["unit_id"],
                    score=0,
                    deductions=[
                        Deduction(
                            dim="tag_error",
                            pts=-100,
                            reason="Tag validation failed",
                        )
                    ],
                    suggested_translation=None,
                    notes="requires manual review",
                )
            )
        else:
            to_score.append(c)

    async def _score_one(c: TranslationCandidate) -> ScoreResult:
        match_patterns = lookup_glossary_or_patterns(c["source"], state.patterns)
        prompt = format_scoring_prompt(
            c["source"],
            c["translation"],
            c["category"],
            c["glossary_base"],
            c["glossary_mods"],
            c["context_result"],
            match_patterns,
        )
        try:
            raw = await llm.ainvoke(
                [
                    SystemMessage(content=system_blocks),
                    HumanMessage(content=prompt),
                ]
            )
            result = ScoreOutputSchema.model_validate(raw)
            score_result: ScoreResult = ScoreResult(
                unit_id=c["unit_id"],
                score=result.score,
                deductions=[
                    Deduction(
                        dim=d.dim,
                        pts=d.pts,
                        reason=d.reason,
                    )
                    for d in result.deductions
                ],
                suggested_translation=result.suggested_translation,
                notes=result.notes,
            )
        except Exception as e:
            logger.warning(f"Scoring failed for {c['source']}: {e}")
            score_result = ScoreResult(
                unit_id=c["unit_id"],
                score=0,
                deductions=[Deduction(dim="parse_error", pts=-100, reason=str(e))],
                suggested_translation=None,
                notes="scorer error",
            )
        logger.info(f"[SCORE] {c['source']}: {score_result['score']}")
        return score_result

    if to_score:
        llm_scores = await asyncio.gather(*[_score_one(c) for c in to_score])
        instant.extend(llm_scores)

    return {"scores": instant}
