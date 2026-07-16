import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from src._share import TEMP_DIR
from src.agent.config import AgentConfigSchema
from src.models.agent import NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient, WeblateUnitSchema


class FetchEmptyOutputSchema(TypedDict):
    to_translate: list[WeblateUnitSchema]
    is_end: bool


@dataclass
class UnitCursor:
    units: list[WeblateUnitSchema]
    offset: int = 0


def _cache_path(component_slug: str, lang: str, q: str | None = None) -> Path:
    return TEMP_DIR / f"{component_slug}_{lang}_{q}.json"


class UnitIterator:
    def __init__(self, client: AsyncWeblateClient):
        self._client = client
        self._cursors: dict[
            tuple[str, str, str | None],
            UnitCursor,
        ] = {}

    async def get_units(
        self,
        component_slug: str,
        lang: str,
        batch_size: int,
        q: str | None = None,
    ) -> tuple[list[WeblateUnitSchema], bool]:
        key = (component_slug, lang, q)
        cursor = self._cursors.get(key)
        if cursor is None:
            units = await self._load_data(component_slug, lang, q)
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
        q: str | None = None,
    ) -> list[WeblateUnitSchema]:
        cursor = self._cursors.get((component_slug, lang, q))
        if cursor is None:
            return []
        return cursor.units[cursor.offset : cursor.offset + batch_size]

    async def _load_data(
        self, component_slug: str, lang: str, q: str | None = None
    ) -> list[WeblateUnitSchema]:
        path = _cache_path(component_slug, lang, q)
        if path.exists():
            return [
                WeblateUnitSchema.model_validate(unit)
                for unit in json.loads(path.read_text("utf-8"))
            ]
        units = await self._client.list_units(
            component_slug=component_slug,
            lang=lang,
            q=q,
        )
        await self._save_data(component_slug, lang, units, q)
        return units

    async def _save_data(
        self,
        component_slug: str,
        lang: str,
        units: list[WeblateUnitSchema],
        q: str | None = None,
    ) -> None:
        path = _cache_path(component_slug, lang, q)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([unit.model_dump() for unit in units], ensure_ascii=False)
        )


async def fetch_empty(
    state: NewAgentStateSchema,
    unit_iterator: UnitIterator,
    agent_config: AgentConfigSchema,
) -> FetchEmptyOutputSchema:
    units, is_end = await unit_iterator.get_units(
        component_slug=agent_config.component_slug,
        lang=agent_config.target_lang,
        batch_size=agent_config.batch_size,
        q="state:empty",
    )
    return {
        "to_translate": units,
        "is_end": is_end,
    }
