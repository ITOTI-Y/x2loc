from __future__ import annotations

import asyncio
from typing import TypedDict

from loguru import logger

from src.agent._share import CONTEXT_COLLECTOR_CONCURRENCY
from src.agent.tools import collect_context_for_term
from src.models.agent import ComponentInfoSchema
from src.services.weblate import AsyncWeblateClient, WeblateUnitSchema


class ContextResultsOutputSchema(TypedDict):
    context_results: dict[int, list[ComponentInfoSchema]]


async def context_collector(
    units: list[WeblateUnitSchema],
    *,
    client: AsyncWeblateClient,
) -> dict[int, list[ComponentInfoSchema]]:
    if not units:
        return {}

    sem = asyncio.Semaphore(CONTEXT_COLLECTOR_CONCURRENCY)

    async def _collect_one(
        unit: WeblateUnitSchema,
    ) -> tuple[int, list[ComponentInfoSchema]]:
        async with sem:
            components = await collect_context_for_term(
                client,
                unit,
            )
        return unit.id, components

    pairs = await asyncio.gather(*[_collect_one(unit) for unit in units])
    logger.success(f"Collected context for {len(pairs)} units")
    return dict(pairs)
