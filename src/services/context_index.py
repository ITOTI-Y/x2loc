"""Cross-component context for translation, served from memory.

A unit's context is where the same source string occurs in other
components, together with that occurrence's neighbours of the same template.
Searching Weblate for it costs one instance-wide query plus one position
query per hit, for every unit; most units have no hit at all, so the
searches alone saturated Weblate (~7 requests/s) and left the LLM waiting.
The index downloads each component's target-language CSV once instead (its
row order is Weblate's `position` order) and answers every lookup locally.
"""

import asyncio
import time
from dataclasses import dataclass

from loguru import logger

from src.agent._share import (
    DEFAULT_NEARBY_RANGE,
    MAX_CONTEXT_COMPONENTS,
    MAX_MATCHES_PER_COMPONENT,
)
from src.models.agent import ComponentInfoSchema
from src.models.weblate import CorpusUnitSchema
from src.services.weblate import AsyncWeblateClient, WeblateAPIError

GLOSSARY_PREFIX = "glossary"


class ContextIndex:
    """Source text -> occurrences, over components held in position order."""

    def __init__(self) -> None:
        self._rows: dict[str, list[CorpusUnitSchema]] = {}
        # source -> slug -> row indices; dicts keep insertion order, which
        # makes the component choice deterministic.
        self._by_source: dict[str, dict[str, list[int]]] = {}

    def put(self, slug: str, rows: list[CorpusUnitSchema]) -> None:
        """Replace one component's rows."""
        for unit in self._rows.get(slug, ()):
            occurrences = self._by_source.get(unit.source)
            if occurrences is not None:
                occurrences.pop(slug, None)
        self._rows[slug] = rows
        for index, unit in enumerate(rows):
            self._by_source.setdefault(unit.source, {}).setdefault(slug, []).append(
                index
            )

    def __contains__(self, slug: str) -> bool:
        return slug in self._rows

    def lookup(self, source: str, *, exclude_slug: str) -> list[ComponentInfoSchema]:
        """Occurrences of `source` outside `exclude_slug`, with neighbours.

        The component being translated is excluded: its neighbours are
        sibling fields of the same template (title next to description), and
        feeding them back misleads the translator and the scorer into
        swapping field contents.
        """
        picked: list[ComponentInfoSchema] = []
        for slug, indices in self._by_source.get(source, {}).items():
            if slug == exclude_slug:
                continue
            rows = self._rows[slug]
            for index in indices[:MAX_MATCHES_PER_COMPONENT]:
                unit = rows[index]
                key = unit.context.split("::")[0]
                window = rows[
                    max(0, index - DEFAULT_NEARBY_RANGE) : index
                    + DEFAULT_NEARBY_RANGE
                    + 1
                ]
                picked.append(
                    ComponentInfoSchema(
                        unit=unit,
                        key=key,
                        slug=slug,
                        position=index + 1,
                        nearby=[row for row in window if key in row.context],
                    )
                )
                if len(picked) == MAX_CONTEXT_COMPONENTS:
                    return picked
        return picked


@dataclass
class _Build:
    started_at: float
    task: asyncio.Task[ContextIndex]


class ContextIndexSource:
    """One context index per process, built on first use.

    A run over already translated mods never asks for context and so never
    pays for the build. Components the run syncs are put in as their files
    are read, and take precedence over the background download. A failed
    build is not reused and its error propagates: translating without the
    context that exists is not an acceptable fallback.
    """

    def __init__(
        self, client: AsyncWeblateClient, *, language: str, ttl_seconds: float
    ) -> None:
        self._client = client
        self._language = language
        self._ttl_seconds = ttl_seconds
        self._index = ContextIndex()
        self._synced: set[str] = set()
        self._build: _Build | None = None

    def update(self, slug: str, rows: list[CorpusUnitSchema]) -> None:
        self._index.put(slug, rows)
        self._synced.add(slug)

    async def index(self) -> ContextIndex:
        build = self._build
        if build is None or not self._reusable(build):
            if build is not None:
                # A rebuild reloads every component, synced ones included.
                self._synced.clear()
            build = self._build = _Build(
                started_at=time.monotonic(),
                task=asyncio.create_task(self._fill()),
            )
        return await asyncio.shield(build.task)

    async def aclose(self) -> None:
        if self._build is not None and not self._build.task.done():
            self._build.task.cancel()
            await asyncio.gather(self._build.task, return_exceptions=True)

    def _reusable(self, build: _Build) -> bool:
        if time.monotonic() - build.started_at > self._ttl_seconds:
            return False
        task = build.task
        return not task.done() or (not task.cancelled() and task.exception() is None)

    async def _fill(self) -> ContextIndex:
        started = time.monotonic()
        slugs = [
            slug
            for slug in await self._client.list_component_slugs()
            if not slug.startswith(GLOSSARY_PREFIX)
        ]
        downloads = await asyncio.gather(
            *(self._download(slug) for slug in slugs if slug not in self._synced)
        )
        for slug, rows in downloads:
            # A component synced while the download ran is fresher.
            if rows is not None and slug not in self._synced:
                self._index.put(slug, rows)
        logger.success(
            "Built context index over {} components in {:.1f}s",
            len(slugs),
            time.monotonic() - started,
        )
        return self._index

    async def _download(self, slug: str) -> tuple[str, list[CorpusUnitSchema] | None]:
        try:
            return slug, await self._client.download_units(slug, self._language)
        except WeblateAPIError as exc:
            # A component without this translation has no context to offer,
            # as the old `language:` filtered search had none from it either.
            if exc.status_code != 404:
                raise
            return slug, None
