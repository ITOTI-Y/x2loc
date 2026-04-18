from __future__ import annotations

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.state import AgentState, TranslationCandidate


def decision_router(state: AgentState, *, agent_config: AgentConfigSchema) -> dict:
    score_map = {s["unit_id"]: s for s in state["scores"]}
    auto: list[TranslationCandidate] = []
    review: list[TranslationCandidate] = []

    for c in state["candidates"]:
        s = score_map.get(c["unit_id"])
        score = s["score"] if s else 0
        if score >= agent_config.auto_approve_threshold:
            auto.append(c)
        else:
            review.append(c)

    logger.info(f"[DECISION] auto={len(auto)}, review={len(review)}")
    return {"auto_batch": auto, "review_batch": review, "needs_review": len(review) > 0}
