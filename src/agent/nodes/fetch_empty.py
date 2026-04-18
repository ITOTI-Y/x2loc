from __future__ import annotations

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.state import AgentState, TranslationUnit
from src.services.weblate import WeblateClient


def fetch_empty(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    page = state["current_page"]
    count, raw_units = client.list_units_page(
        agent_config.component_slug,
        agent_config.target_lang,
        q="state:empty",
        page=page,
        page_size=agent_config.batch_size,
    )

    skip_ids = state["skip_ids"]
    units: list[TranslationUnit] = []
    for u in raw_units:
        if u["id"] in skip_ids:
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

    should_continue = count > 0 and len(units) > 0
    logger.info(f"Fetched {len(units)} units (page {page}), {count} remaining")
    return {
        "current_units": units,
        "remaining_count": count,
        "current_page": page + 1,
        "should_continue": should_continue,
    }
