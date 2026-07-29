from dataclasses import dataclass
from typing import TypedDict

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.models.agent import NewAgentStateSchema
from src.models.weblate import WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient


class FetchEmptyOutputSchema(TypedDict):
    to_translate: list[WeblateUnitSchema]
    is_end: bool


@dataclass
class UnitCursor:
    units: list[WeblateUnitSchema]
    offset: int = 0


class UnitIterator:
    """Per-run snapshot iterator over a stateful Weblate query.

    The snapshot lives only in this iterator's memory: persisting it would
    let a later run in the same process act on stale emptiness and overwrite
    targets Weblate already holds.
    """

    def __init__(self, client: AsyncWeblateClient):
        self._client = client
        self._cursors: dict[tuple[str, str, str], UnitCursor] = {}

    async def get_units(
        self,
        component_slug: str,
        lang: str,
        batch_size: int,
        q: str = "",
    ) -> tuple[list[WeblateUnitSchema], bool]:
        key = (component_slug, lang, q)
        cursor = self._cursors.get(key)
        if cursor is None:
            units = await self._client.list_units(component_slug, lang, q=q)
            cursor = self._cursors[key] = UnitCursor(units)

        start = cursor.offset
        cursor.offset = start + batch_size

        batch = cursor.units[start : start + batch_size]

        if not batch:
            del self._cursors[key]
            return [], True
        return batch, False

    def peek_units(
        self,
        component_slug: str,
        lang: str,
        batch_size: int,
        q: str = "",
    ) -> list[WeblateUnitSchema]:
        cursor = self._cursors.get((component_slug, lang, q))
        if cursor is None:
            return []
        return cursor.units[cursor.offset : cursor.offset + batch_size]


async def fetch_empty(
    state: NewAgentStateSchema,
    unit_iterator: UnitIterator,
    agent_config: AgentConfigSchema,
) -> FetchEmptyOutputSchema:
    units, is_end = await unit_iterator.get_units(
        component_slug=state.component_slug,
        lang=agent_config.target_lang,
        batch_size=agent_config.batch_size,
        q="state:empty",
    )
    logger.success(f"Fetched {len(units)} empty units")
    return {
        "to_translate": units,
        "is_end": is_end,
    }
