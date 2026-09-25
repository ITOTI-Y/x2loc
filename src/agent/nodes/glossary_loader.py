import asyncio
import time
from dataclasses import dataclass
from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.nodes.pattern_extractor import mine_glossary_patterns
from src.core.glossary import group_units
from src.models.agent import PatternSchema
from src.models.weblate import WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient


class GlossaryLoaderOutputSchema(TypedDict):
    base_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    mods_glossary: dict[str, tuple[WeblateUnitSchema, ...]]
    patterns: dict[str, tuple[PatternSchema, ...]]


@dataclass(frozen=True)
class _Snapshot:
    started_at: float
    units: asyncio.Task[list[WeblateUnitSchema]]


class GlossaryCache:
    """Glossary snapshots shared by every job of one process.

    A full glossary read takes minutes on a slow Weblate, so each glossary
    is fetched once and reused until `ttl_seconds` pass or `invalidate`
    marks it stale; the pipeline invalidates the custom glossary right after
    appending terms to it, so the next job sees them. Reads are
    single-flight and shielded: a cancelled job never aborts a read that
    other jobs are waiting on, and a failed read is not reused.
    """

    def __init__(self, client: AsyncWeblateClient, *, ttl_seconds: float) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._snapshots: dict[tuple[str, str], _Snapshot] = {}

    async def load(self, config: ConfigSchema) -> GlossaryLoaderOutputSchema:
        """Index the three glossaries and mine translation patterns from them.

        `mods` and `custom` are merged: both are mod-scoped terminology and
        the translator consults them as one table.
        """
        base, mods, custom = await asyncio.gather(
            *(
                self._units(slug, config.target_lang)
                for slug in (
                    config.base_glossary_slug,
                    config.mods_glossary_slug,
                    config.custom_glossary_slug,
                )
            )
        )
        base_glossary = group_units(base)
        mods_glossary = group_units([*mods, *custom])
        patterns = await asyncio.to_thread(
            mine_glossary_patterns, base_glossary, mods_glossary
        )
        logger.success(
            "Loaded glossaries: {} base + {} mods + {} custom; mined {} patterns",
            len(base),
            len(mods),
            len(custom),
            len(patterns),
        )
        return {
            "base_glossary": base_glossary,
            "mods_glossary": mods_glossary,
            "patterns": patterns,
        }

    async def aclose(self) -> None:
        """Cancel reads still in flight; the owner calls this before closing
        the Weblate client they use."""
        pending = [s.units for s in self._snapshots.values() if not s.units.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._snapshots.clear()

    def invalidate(self, slug: str) -> None:
        self._snapshots = {
            key: snapshot for key, snapshot in self._snapshots.items() if key[0] != slug
        }

    async def _units(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        key = (slug, language)
        snapshot = self._snapshots.get(key)
        if snapshot is None or not self._reusable(snapshot):
            snapshot = _Snapshot(
                started_at=time.monotonic(),
                units=asyncio.create_task(
                    self._client.list_units(slug, language, q="state:translated")
                ),
            )
            self._snapshots[key] = snapshot
        return await asyncio.shield(snapshot.units)

    def _reusable(self, snapshot: _Snapshot) -> bool:
        if time.monotonic() - snapshot.started_at > self._ttl_seconds:
            return False
        task = snapshot.units
        return not task.done() or (not task.cancelled() and task.exception() is None)
