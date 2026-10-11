import asyncio
import re
from typing import Any

from src.agent.config import ConfigSchema
from src.agent.nodes.translator import translator
from src.models.agent import (
    NewAgentStateSchema,
    TranslationBatchOutputSchema,
    TranslationItemSchema,
)
from src.models.weblate import WeblateUnitSchema


class FakeAgent:
    """Answers every item with "模型译文" and records the prompts it saw."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = payload["messages"][0]["content"]
        self.prompts.append(content)
        ids = [int(i) for i in re.findall(r"## Item (\d+)", content)]
        return {
            "structured_response": TranslationBatchOutputSchema(
                results=[TranslationItemSchema(id=i, result="模型译文") for i in ids]
            )
        }


def _unit(unit_id: int, context: str) -> WeblateUnitSchema:
    return WeblateUnitSchema(
        id=unit_id,
        language_code="zh_Hans",
        source=f"Source {context}",
        target="",
        context=context,
    )


def _state(attempts: int) -> NewAgentStateSchema:
    units = [_unit(1, "shipped"), _unit(2, "missing")]
    return NewAgentStateSchema(
        references={"shipped": "模组译文"},
        to_translate=units,
        context_results={unit.id: [] for unit in units},
        attempts=attempts,
    )


async def test_first_round_reviews_the_mods_translation_as_it_is(
    agent_config: ConfigSchema,
) -> None:
    agent = FakeAgent()

    result = await translator(
        _state(attempts=0),
        agent_config=agent_config,
        llm_slots=asyncio.Semaphore(1),
        agent=agent,  # ty: ignore[invalid-argument-type]
    )

    assert {c.id: c.translated for c in result["candidates"]} == {
        1: "模组译文",
        2: "模型译文",
    }
    assert all("## Item 1" not in prompt for prompt in agent.prompts)


async def test_retry_round_shows_the_mods_translation_as_a_reference(
    agent_config: ConfigSchema,
) -> None:
    agent = FakeAgent()

    result = await translator(
        _state(attempts=1),
        agent_config=agent_config,
        llm_slots=asyncio.Semaphore(1),
        agent=agent,  # ty: ignore[invalid-argument-type]
    )

    assert {c.id: c.translated for c in result["candidates"]} == {
        1: "模型译文",
        2: "模型译文",
    }
    assert "模组译文" in "".join(agent.prompts)
