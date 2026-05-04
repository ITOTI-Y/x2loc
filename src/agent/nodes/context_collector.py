from __future__ import annotations

import asyncio

from src.agent._share import CONTEXT_COLLECTOR_CONCURRENCY
from src.agent.config import AgentConfigSchema
from src.agent.state import AgentState, ContextResult, TranslationUnit
from src.agent.tools import collect_context_for_term
from src.services.weblate import WeblateClient


async def context_collector(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    sem = asyncio.Semaphore(CONTEXT_COLLECTOR_CONCURRENCY)

    async def _collect_one(unit: TranslationUnit) -> ContextResult:
        async with sem:
            ctx = await asyncio.to_thread(
                collect_context_for_term,
                client,
                unit["source"],
                lang=agent_config.target_lang,
            )
        return {
            "unit_id": unit["id"],
            "mod_component": ctx["mod_component"],
            "translated_percent": ctx["translated_percent"],
            "nearby": ctx["nearby"],
        }

    results = await asyncio.gather(
        *[_collect_one(unit) for unit in state["current_units"]]
    )
    return {"context_results": list(results)}
