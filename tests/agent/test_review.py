from src.agent.config import ConfigSchema
from src.agent.review import ThresholdReview
from src.models.agent import (
    NewAgentStateSchema,
    ScoreResultSchema,
    TranslationUnitSchema,
)
from src.models.weblate import WeblateUnitSchema


def scored_unit(unit_id: int, *, tag_valid: bool) -> TranslationUnitSchema:
    original = WeblateUnitSchema(
        id=unit_id,
        language_code="zh_Hans",
        source="OK",
        target="",
        context=f"key-{unit_id}",
    )
    return TranslationUnitSchema(
        id=unit_id,
        source="OK",
        translated="确定",
        key=f"key-{unit_id}",
        context=[],
        category="",
        pattern_matched=False,
        glossary_base=[],
        glossary_mods=[],
        tag_valid=tag_valid,
        original_unit=original,
        patterns=[],
    )


async def test_exhausted_gate_skips_failures_and_keeps_passes(
    agent_config: ConfigSchema,
) -> None:
    config = agent_config.model_copy(update={"auto_approve_threshold": 0})
    state = NewAgentStateSchema(
        scores=[scored_unit(1, tag_valid=True), scored_unit(2, tag_valid=False)],
        attempts=config.max_translation_attempts - 1,
    )

    result = await ThresholdReview()(state, agent_config=config)

    actions = {decision.unit_id: decision.action for decision in result["decisions"]}
    assert actions == {1: "approve", 2: "skip"}
    assert not result["retry_pending"]


def with_score(
    unit: TranslationUnitSchema, score: int, translated: str
) -> TranslationUnitSchema:
    return unit.model_copy(
        update={
            "translated": translated,
            "score_result": ScoreResultSchema(score=score),
        }
    )


async def test_exhausted_gate_writes_best_candidate_as_needs_editing(
    agent_config: ConfigSchema,
) -> None:
    review = ThresholdReview()
    state = NewAgentStateSchema(
        scores=[with_score(scored_unit(1, tag_valid=True), 55, "最好")],
        attempts=agent_config.max_translation_attempts - 2,
    )
    retry = await review(state, agent_config=agent_config)
    assert retry["retry_pending"]
    assert retry["best_candidates"] == {1: (55, "最好")}

    last = NewAgentStateSchema(
        scores=[with_score(scored_unit(1, tag_valid=True), 45, "较差")],
        attempts=retry["attempts"],
        best_candidates=retry["best_candidates"],
    )
    result = await review(last, agent_config=agent_config)

    assert [(d.action, d.translation) for d in result["decisions"]] == [
        ("needs_editing", "最好")
    ]
    assert result["best_candidates"] == {}


async def test_tag_invalid_candidates_are_never_kept(
    agent_config: ConfigSchema,
) -> None:
    state = NewAgentStateSchema(
        scores=[with_score(scored_unit(1, tag_valid=False), 90, "坏标签")],
        attempts=agent_config.max_translation_attempts - 1,
    )

    result = await ThresholdReview()(state, agent_config=agent_config)

    assert [d.action for d in result["decisions"]] == ["skip"]
