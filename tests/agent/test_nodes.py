import asyncio

import pytest

import src.agent.nodes as nodes_module
from src.agent.config import ConfigSchema
from src.agent.nodes import WorkflowNodes
from src.agent.review import ThresholdReview
from src.models.agent import NewAgentStateSchema
from src.services.glossary import GlossarySnapshots
from src.services.weblate import AsyncWeblateClient


async def _agents_used(
    config: ConfigSchema, monkeypatch: pytest.MonkeyPatch
) -> list[object]:
    used: list[object] = []

    async def fake_translator(_state, *, agent_config, llm_slots, agent):
        used.append(agent)
        return {"candidates": []}

    monkeypatch.setattr(nodes_module, "translator", fake_translator)
    client = AsyncWeblateClient(config.weblate)
    nodes = WorkflowNodes(
        client,
        config,
        review=ThresholdReview(),
        glossaries=GlossarySnapshots(client, ttl_seconds=60),
        llm_slots=asyncio.Semaphore(1),
    )
    await nodes.translator(NewAgentStateSchema(attempts=0))
    await nodes.translator(NewAgentStateSchema(attempts=1))
    await nodes.aclose()
    return used


async def test_retry_round_uses_retry_model(
    agent_config: ConfigSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = agent_config.model_copy(
        update={"retry_translation_model_name": "strong-model"}
    )
    used = await _agents_used(config, monkeypatch)
    first, retry = used
    assert first is not retry


async def test_empty_retry_model_reuses_translation_agent(
    agent_config: ConfigSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    used = await _agents_used(agent_config, monkeypatch)
    first, retry = used
    assert first is retry
    assert agent_config.effective_retry_translation_model == "test-model"
