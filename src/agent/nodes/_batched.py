from __future__ import annotations

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
    max_concurrency: int,
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
        max_concurrency: Requests in flight.
        label: Node name for log lines.

    Returns:
        Structured result per answered unit id.
    """
    chunks = [
        items[start : start + units_per_request]
        for start in range(0, len(items), units_per_request)
    ]
    responses = await agent.abatch(
        [
            {"messages": [{"role": "user", "content": format_batch(chunk)}]}
            for chunk in chunks
        ],
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
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
