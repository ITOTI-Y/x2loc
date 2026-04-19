from __future__ import annotations

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.state import AgentState, TranslationUnit
from src.services.weblate import WeblateAPIError, WeblateClient


def fetch_empty(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    page = state["current_page"]
    skip_set = set(state["skip_ids"])
    target = agent_config.batch_size
    units: list[TranslationUnit] = []
    count = 0

    while len(units) < target:
        try:
            count, raw_units = client.list_units_page(
                agent_config.component_slug,
                agent_config.target_lang,
                q="state:empty",
                page=page,
                page_size=target,
            )
        except WeblateAPIError as e:
            if e.status != 404:
                raise
            logger.warning(f"Page {page:d} out of range, pool shrank")
            page = 1
            break

        if not raw_units:
            break

        for u in raw_units:
            if u["id"] in skip_set:
                continue
            ctx = u["context"]
            category = ctx.split("::")[-1] if "::" in ctx else None
            units.append(
                {
                    "id": u["id"],
                    "position": u["position"],
                    "source": u["source"][0],
                    "context": ctx,
                    "category": category,
                }
            )
            if len(units) >= target:
                break

        page += 1

        if len(raw_units) < target:
            break

    should_continue = count > 0 and len(units) > 0
    logger.info(
        f"Fetched {len(units):d} units (up to page {page - 1:d}), {count:d} total empty"
    )
    return {
        "current_units": units,
        "remaining_count": count,
        "current_page": page,
        "should_continue": should_continue,
    }
