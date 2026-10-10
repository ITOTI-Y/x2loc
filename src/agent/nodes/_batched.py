from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import Runnable
from loguru import logger

from src.agent.llm import raise_if_fatal_llm_error
from src.agent.prompts import format_batch
from src.models.agent import AgentInputSchema


async def invoke_batched(
    agent: Runnable[AgentInputSchema, Any],
    items: list[tuple[int, str]],
    *,
    units_per_request: int,
    slots: asyncio.Semaphore,
    label: str,
) -> dict[int, Any]:
    """Send per-item prompts `units_per_request` at a time; map results by id.

    A failed request or an id the model left out is simply absent from the
    result, so callers treat it like any empty answer and the quality gate
    retries it. Results carrying an id that was not asked for are dropped.

    Args:
        agent: Structured-output agent whose response has `results`, each
            entry carrying the `id` of its item.
        items: `(unit id, single-item prompt)` pairs.
        units_per_request: Items packed into one request.
        slots: Bounds LLM requests in flight; shared by every node, component
            and job that draws on the same LLM budget.
        label: Node name for log lines.

    Returns:
        Structured result per answered unit id.
    """
    chunks = [
        items[start : start + units_per_request]
        for start in range(0, len(items), units_per_request)
    ]

    async def send(chunk: list[tuple[int, str]]) -> Any:
        async with slots:
            return await agent.ainvoke(
                {"messages": [{"role": "user", "content": format_batch(chunk)}]}
            )

    responses = await asyncio.gather(
        *(send(chunk) for chunk in chunks), return_exceptions=True
    )
    results: dict[int, Any] = {}
    for chunk, response in zip(chunks, responses, strict=True):
        asked = {item_id for item_id, _prompt in chunk}
        if isinstance(response, BaseException):
            raise_if_fatal_llm_error(response)
            logger.warning(
                "{} request for {} units failed: {!r}", label, len(chunk), response
            )
            continue
        structured = response.get("structured_response") if response else None
        answered = {
            result.id: result
            for result in getattr(structured, "results", [])
            if result.id in asked
        }
        if len(answered) < len(asked):
            logger.warning(
                "{} request answered {}/{} units", label, len(answered), len(asked)
            )
        results.update(answered)
    return results
