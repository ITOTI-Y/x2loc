from __future__ import annotations

from typing import Protocol, TypedDict

from langgraph.types import interrupt
from loguru import logger

from src.agent.config import ConfigSchema
from src.models.agent import (
    NewAgentStateSchema,
    ReviewDecisionSchema,
    TranslationUnitSchema,
)
from src.models.weblate import WeblateUnitSchema


class ReviewOutputSchema(TypedDict):
    decisions: list[ReviewDecisionSchema]
    accepted_decisions: list[ReviewDecisionSchema]
    to_translate: list[WeblateUnitSchema]
    quality_feedback: dict[int, str]
    attempts: int
    retry_pending: bool
    best_candidates: dict[int, tuple[int, str]]


class ReviewPolicy(Protocol):
    """Turns scored candidates into accept/skip decisions.

    `extracts_patterns` selects whether the graph runs `pattern_extractor`
    after upload to add templates from this session's decisions on top of
    the glossary-derived ones. Only human-approved translations may feed
    it, so the automatic policy leaves it out.
    """

    extracts_patterns: bool

    async def __call__(
        self, state: NewAgentStateSchema, *, agent_config: ConfigSchema
    ) -> ReviewOutputSchema: ...


class InterruptReview:
    """Hand the scored batch to a human through LangGraph's interrupt."""

    extracts_patterns = True

    async def __call__(
        self, state: NewAgentStateSchema, *, agent_config: ConfigSchema
    ) -> ReviewOutputSchema:
        decisions: list[ReviewDecisionSchema] = interrupt(state.scores)
        return {
            "decisions": decisions,
            "accepted_decisions": [],
            "to_translate": [],
            "quality_feedback": {},
            "attempts": 0,
            "retry_pending": False,
            "best_candidates": {},
        }


class ThresholdReview:
    """Accept on a valid tag set and a score at or above the threshold.

    Failures are fed back into the next translation round. Once
    `max_translation_attempts` rounds are spent, each remaining failure gets
    its highest-scoring tag-valid candidate from any round, written as
    "needs editing" for a human to review; that state also keeps it out of
    the next run's empty-unit fetch. A unit with no tag-valid candidate is
    skipped and stays empty: a broken tag set would fail the overlay check.
    """

    extracts_patterns = False

    async def __call__(
        self, state: NewAgentStateSchema, *, agent_config: ConfigSchema
    ) -> ReviewOutputSchema:
        accepted = list(state.accepted_decisions)
        failed: list[TranslationUnitSchema] = []
        for unit in state.scores:
            score = unit.score_result.score if unit.score_result else 0
            if (
                unit.translated.strip()
                and unit.tag_valid
                and score >= agent_config.auto_approve_threshold
            ):
                accepted.append(
                    ReviewDecisionSchema(
                        unit_id=unit.id, action="approve", translation=unit.translated
                    )
                )
            else:
                failed.append(unit)

        best = dict(state.best_candidates)
        for unit in failed:
            score = unit.score_result.score if unit.score_result else 0
            if unit.translated.strip() and unit.tag_valid:
                previous = best.get(unit.id)
                if previous is None or score > previous[0]:
                    best[unit.id] = (score, unit.translated)

        if not failed:
            return {
                "decisions": accepted,
                "accepted_decisions": [],
                "to_translate": [],
                "quality_feedback": {},
                "attempts": 0,
                "retry_pending": False,
                "best_candidates": {},
            }

        attempts = state.attempts + 1
        if attempts >= agent_config.max_translation_attempts:
            exhausted: list[ReviewDecisionSchema] = []
            for unit in failed:
                score = unit.score_result.score if unit.score_result else 0
                notes = unit.score_result.notes if unit.score_result else ""
                candidate = best.get(unit.id)
                logger.warning(
                    "Quality gate exhausted for unit {} [{}], {}: score={} "
                    "tag_valid={} notes={!r} source={!r} last_translation={!r}",
                    unit.id,
                    unit.key,
                    (
                        f"writing best candidate (score {candidate[0]}) "
                        "as needs editing"
                        if candidate
                        else "leaving it empty"
                    ),
                    score,
                    unit.tag_valid,
                    notes,
                    unit.source[:120],
                    unit.translated[:120],
                )
                exhausted.append(
                    ReviewDecisionSchema(
                        unit_id=unit.id,
                        action="needs_editing",
                        translation=candidate[1],
                    )
                    if candidate
                    else ReviewDecisionSchema(
                        unit_id=unit.id, action="skip", translation=""
                    )
                )
            return {
                "decisions": accepted + exhausted,
                "accepted_decisions": [],
                "to_translate": [],
                "quality_feedback": {},
                "attempts": 0,
                "retry_pending": False,
                "best_candidates": {},
            }

        logger.warning(
            "Quality gate rejected {} units; retry {}/{}",
            len(failed),
            attempts,
            agent_config.max_translation_attempts,
        )
        return {
            "decisions": [],
            "accepted_decisions": accepted,
            "to_translate": [unit.original_unit for unit in failed],
            "quality_feedback": {unit.id: _feedback(unit) for unit in failed},
            "attempts": attempts,
            "retry_pending": True,
            "best_candidates": best,
        }


def _feedback(unit: TranslationUnitSchema) -> str:
    if unit.score_result is None:
        return "No structured score was returned. Produce a complete valid translation."
    deductions = "; ".join(
        f"{item.dim}: -{item.pts} {item.reason}"
        for item in unit.score_result.deductions
    )
    return (
        f"Previous score: {unit.score_result.score}. "
        f"Tag valid: {unit.tag_valid}. Deductions: {deductions}. "
        f"Suggested translation: {unit.score_result.suggested_translation}"
    )
