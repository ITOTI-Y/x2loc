from src.agent.config import ConfigSchema
from src.agent.review import ThresholdReview
from src.models.agent import NewAgentStateSchema, TranslationUnitSchema
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
