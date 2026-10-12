"""Reads of Weblate shared by every store; none of them needs a token."""

import asyncio

from src.models.weblate import CorpusUnitSchema
from src.models.workshop import AssetWorkSchema
from src.services.weblate import AsyncWeblateClient


async def true_sources(
    client: AsyncWeblateClient,
    work: AssetWorkSchema,
    snapshot: list[CorpusUnitSchema],
) -> list[CorpusUnitSchema]:
    """The snapshot with each unit's source as Weblate's template holds it.

    Weblate's CSV export keeps a unit's source column at the text the unit
    was created with, even after the source is edited, while the
    source-language file's target column always holds the current one.
    That file is read only when the snapshot disagrees with the parsed
    sources, which is rare.
    """
    if all(
        work.expected.get(unit.context, unit.source) == unit.source for unit in snapshot
    ):
        return snapshot
    sources = {
        unit.context: unit.target
        for unit in await client.download_units(
            work.asset.component_slug, client.config.source_language
        )
    }
    return [
        unit.model_copy(update={"source": sources.get(unit.context, unit.source)})
        for unit in snapshot
    ]


class NeedsEditing:
    """Which units Weblate holds as "needs editing", per language.

    One search per language serves the whole run. A store reports the
    states it writes itself through `record`, which wins over the search:
    a write may land while the search is still in flight.
    """

    def __init__(self, client: AsyncWeblateClient) -> None:
        self._client = client
        self._loads: dict[str, asyncio.Task[set[tuple[str, str]]]] = {}
        self._recorded: dict[tuple[str, str, str], bool] = {}

    def record(self, language: str, slug: str, context: str, *, fuzzy: bool) -> None:
        self._recorded[language, slug, context] = fuzzy

    async def contains(self, language: str, slug: str, context: str) -> bool:
        recorded = self._recorded.get((language, slug, context))
        if recorded is not None:
            return recorded
        return (slug, context) in await self._load(language)

    def _load(self, language: str) -> asyncio.Task[set[tuple[str, str]]]:
        # No await between the lookup and the assignment: concurrent
        # components share one search.
        task = self._loads.get(language)
        if task is None:
            task = self._loads[language] = asyncio.create_task(self._search(language))
        return task

    async def _search(self, language: str) -> set[tuple[str, str]]:
        units = await self._client.search_units(
            f"project:{self._client.config.project_slug} AND "
            f"language:{language} AND state:needs-editing"
        )
        # A unit's translation URL ends in .../<component>/<language>/.
        return {
            (unit.translation.rstrip("/").split("/")[-2], unit.context)
            for unit in units
        }

    async def aclose(self) -> None:
        for task in self._loads.values():
            task.cancel()
        await asyncio.gather(*self._loads.values(), return_exceptions=True)
