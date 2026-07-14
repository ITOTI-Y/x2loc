from __future__ import annotations

import asyncio
from typing import TypedDict

from src.agent._share import CONTEXT_COLLECTOR_CONCURRENCY
from src.agent.config import AgentConfigSchema
from src.agent.tools import collect_context_for_term
from src.models.agent import ComponentInfoSchema, NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient, WeblateUnitSchema


class ContextResultsOutputSchema(TypedDict):
    context_results: dict[int, list[ComponentInfoSchema]]


async def context_collector(
    state: NewAgentStateSchema,
    *,
    client: AsyncWeblateClient,
    agent_config: AgentConfigSchema,
) -> ContextResultsOutputSchema:
    sem = asyncio.Semaphore(CONTEXT_COLLECTOR_CONCURRENCY)

    async def _collect_one(
        unit: WeblateUnitSchema,
    ) -> tuple[int, list[ComponentInfoSchema]]:
        async with sem:
            components = await collect_context_for_term(
                client,
                unit.source,
                agent_config.target_lang,
            )
        return unit.id, components

    pairs = await asyncio.gather(*[_collect_one(unit) for unit in state.to_translate])
    return {"context_results": dict(pairs)}
