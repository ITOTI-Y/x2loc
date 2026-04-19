from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.llm import ScoreOutputSchema, build_scorer_llm
from src.agent.prompts import SCORING_SYSTEM, format_scoring_prompt
from src.agent.state import AgentState, Deduction, ScoreResult, TranslationCandidate


async def scorer(state: AgentState, *, agent_config: AgentConfigSchema) -> dict:
    scorer_llm = build_scorer_llm(agent_config)
    instant: list[ScoreResult] = []
    to_score: list[TranslationCandidate] = []

    for c in state["candidates"]:
        if c["pattern_matched"]:
            instant.append(
                {
                    "unit_id": c["unit_id"],
                    "score": 100,
                    "deductions": [],
                    "suggested_alternative": None,
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
                    suggested_alternative=None,
                    notes="requires manual review",
                )
            )
        else:
            to_score.append(c)

    async def _score_one(
        c: TranslationCandidate, *, agent_config: AgentConfigSchema
    ) -> ScoreResult:
        match_patterns = []
        for single_word in c["source"].split():
            match_patterns.extend(i for i in state["session_patterns"] if single_word.lower() in i["src_pattern"].lower())
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
            raw = await scorer_llm.ainvoke(
                [
                    SystemMessage(
                        content=SCORING_SYSTEM.format(
                            target_lang=agent_config.target_lang
                        )
                    ),
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
                suggested_alternative=result.suggested_alternative,
                notes=result.notes,
            )
        except Exception as e:
            logger.warning(f"Scoring failed for {c['source']}: {e}")
            score_result = ScoreResult(
                unit_id=c["unit_id"],
                score=0,
                deductions=[Deduction(dim="parse_error", pts=-100, reason=str(e))],
                suggested_alternative=None,
                notes="scorer error",
            )
        logger.info(f"[SCORE] {c['source']}: {score_result['score']}")
        return score_result

    if to_score:
        llm_scores = await asyncio.gather(
            *[_score_one(c, agent_config=agent_config) for c in to_score]
        )
        instant.extend(llm_scores)

    return {"scores": instant}
