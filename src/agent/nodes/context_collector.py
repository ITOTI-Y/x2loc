from __future__ import annotations

import asyncio

from src.agent._share import CONTEXT_COLLECTOR_CONCURRENCY
from src.agent.config import AgentConfigSchema
from src.agent.state import ContextResult
from src.agent.tools import collect_context_for_term
from src.models.agent import NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient, WeblateUnitSchema


async def context_collector(
    state: NewAgentStateSchema, *, client: AsyncWeblateClient, agent_config: AgentConfigSchema
) -> dict:
    sem = asyncio.Semaphore(CONTEXT_COLLECTOR_CONCURRENCY)

    async def _collect_one(unit: WeblateUnitSchema) -> ContextResult:
        async with sem:
            ctx = await collect_context_for_term(
                client,
                unit.source,
                agent_config.target_lang,
            )
        return {
            "unit_id": unit.id,
            "mod_component": ctx["mod_component"],
            "translated_percent": ctx["translated_percent"],
            "nearby": ctx["nearby"],
        }

    results = await asyncio.gather(
        *[_collect_one(unit) for unit in state.to_translate]
    )
    return {"context_results": list(results)}
