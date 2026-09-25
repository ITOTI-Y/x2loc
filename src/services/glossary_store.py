"""Persistent local glossary snapshots with deferred custom-glossary writes.

Batch runs read each glossary from disk, pull only what changed in Weblate
since the last sync, and queue newly mined custom terms locally until the
run ends. Layout under `root` (one pair of files per glossary/language):

- `{slug}.{language}.json`: `StoredGlossarySchema`, the translated units.
- `{slug}.{language}.pending.jsonl`: one `{"source", "target"}` per line,
  terms mined by this or an interrupted earlier run, not yet in Weblate.
"""

import asyncio
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from loguru import logger
from pydantic import Field

from src.core.glossary import term_context
from src.models._share import BaseSchema
from src.models.glossary import GlossaryTerm
from src.models.weblate import WeblateUnitSchema
from src.services.glossary import existing_pairs, publish_pairs, term_pairs
from src.services.weblate import AsyncWeblateClient

WEBLATE_STATE_TRANSLATED: Final = 20
CHANGED_LOOKBACK: Final = timedelta(days=1)
FULL_SYNC_INTERVAL: Final = timedelta(days=1)


class StoredGlossarySchema(BaseSchema):
    slug: str
    language: str
    full_synced_at: datetime
    synced_at: datetime
    units: list[WeblateUnitSchema] = Field(default_factory=list)


class LocalGlossaryStore:
    """Glossary units kept on disk and refreshed incrementally from Weblate.

    Each glossary is synced at most once per `refresh_seconds` within a
    process; pending custom terms are served as local units (negative ids)
    so later jobs of the same run already see them.
    """

    def __init__(
        self, client: AsyncWeblateClient, root: Path, *, refresh_seconds: float
    ) -> None:
        self._client = client
        self._root = root
        self._refresh_seconds = refresh_seconds
        self._stored: dict[tuple[str, str], StoredGlossarySchema] = {}
        self._refreshed_at: dict[tuple[str, str], float] = {}
        self._pending: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def units(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        key = (slug, language)
        async with self._locks.setdefault(key, asyncio.Lock()):
            refreshed = self._refreshed_at.get(key)
            if (
                refreshed is None
                or time.monotonic() - refreshed > self._refresh_seconds
            ):
                self._stored[key] = await self._sync(slug, language)
                self._refreshed_at[key] = time.monotonic()
        local = [
            WeblateUnitSchema(
                id=-(index + 1),
                language_code=language,
                source=source,
                target=target,
                context=term_context(source, target),
                state=WEBLATE_STATE_TRANSLATED,
            )
            for index, (source, target) in enumerate(self.pending(slug, language))
        ]
        return [*self._stored[key].units, *local]

    def pending(self, slug: str, language: str) -> list[tuple[str, str]]:
        key = (slug, language)
        if key not in self._pending:
            path = self._pending_path(slug, language)
            lines = path.read_text("utf-8").splitlines() if path.exists() else []
            self._pending[key] = [
                (row["source"], row["target"]) for row in map(json.loads, lines) if row
            ]
        return self._pending[key]

    def add_pending(
        self, slug: str, language: str, pairs: Iterable[tuple[str, str]]
    ) -> None:
        """Append to the on-disk queue first, so a crash cannot lose terms."""
        new = list(pairs)
        if not new:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        with self._pending_path(slug, language).open("a", encoding="utf-8") as f:
            for source, target in new:
                f.write(
                    json.dumps({"source": source, "target": target}, ensure_ascii=False)
                )
                f.write("\n")
        self.pending(slug, language).extend(new)

    def clear_pending(self, slug: str, language: str) -> None:
        self._pending_path(slug, language).unlink(missing_ok=True)
        self._pending[(slug, language)] = []
        self._refreshed_at.pop((slug, language), None)

    async def aclose(self) -> None:
        """Nothing runs in the background; present for the source protocol."""

    async def _sync(self, slug: str, language: str) -> StoredGlossarySchema:
        path = self._snapshot_path(slug, language)
        stored = await asyncio.to_thread(self._read, path) if path.exists() else None
        started = datetime.now(UTC)
        if stored is None or started - stored.full_synced_at > FULL_SYNC_INTERVAL:
            units = await self._client.list_units(slug, language, q="state:translated")
            stored = StoredGlossarySchema(
                slug=slug,
                language=language,
                full_synced_at=started,
                synced_at=started,
                units=units,
            )
            logger.info("Glossary {} full sync: {} units", slug, len(units))
        else:
            since = (stored.synced_at - CHANGED_LOOKBACK).strftime("%Y-%m-%d %H:%M")
            changed = await self._client.list_units(
                slug, language, q=f'changed:>="{since}"'
            )
            by_id = {unit.id: unit for unit in stored.units}
            for unit in changed:
                if unit.state >= WEBLATE_STATE_TRANSLATED and unit.target.strip():
                    by_id[unit.id] = unit
                else:
                    by_id.pop(unit.id, None)
            stored = stored.model_copy(
                update={"synced_at": started, "units": list(by_id.values())}
            )
            logger.info(
                "Glossary {} incremental sync: {} changed, {} units",
                slug,
                len(changed),
                len(stored.units),
            )
        await asyncio.to_thread(self._write, path, stored)
        return stored

    def _snapshot_path(self, slug: str, language: str) -> Path:
        return self._root / f"{slug}.{language}.json"

    def _pending_path(self, slug: str, language: str) -> Path:
        return self._root / f"{slug}.{language}.pending.jsonl"

    @staticmethod
    def _read(path: Path) -> StoredGlossarySchema:
        return StoredGlossarySchema.model_validate_json(path.read_text("utf-8"))

    @staticmethod
    def _write(path: Path, stored: StoredGlossarySchema) -> None:
        """Atomic replace: an interrupted write must not truncate the snapshot."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(stored.model_dump_json(), "utf-8")
        temporary.replace(path)


class DeferredGlossaryWriter:
    """Queues new custom-glossary terms locally; `flush` publishes them."""

    def __init__(
        self,
        store: LocalGlossaryStore,
        *,
        component_slug: str,
        target_lang: str,
    ) -> None:
        self._store = store
        self._component_slug = component_slug
        self._target_lang = target_lang
        self._lock = asyncio.Lock()

    async def write(self, terms: list[GlossaryTerm]) -> tuple[int, int]:
        pairs = term_pairs(terms)
        if not pairs:
            return 0, 0
        async with self._lock:
            current = await self._store.units(self._component_slug, self._target_lang)
            existing = existing_pairs(current)
            new_pairs = [pair for pair in pairs if pair not in existing]
            self._store.add_pending(self._component_slug, self._target_lang, new_pairs)
        return len(new_pairs), len(pairs) - len(new_pairs)

    async def flush(self, client: AsyncWeblateClient) -> int:
        """Publish every queued term; the queue survives a failed attempt."""
        async with self._lock:
            pairs = list(self._store.pending(self._component_slug, self._target_lang))
            if not pairs:
                return 0
            await publish_pairs(
                client,
                component_slug=self._component_slug,
                target_lang=self._target_lang,
                pairs=pairs,
            )
            self._store.clear_pending(self._component_slug, self._target_lang)
        return len(pairs)
