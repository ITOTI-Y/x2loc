"""Translations kept on this machine, over a read-only view of Weblate.

Weblate is only read, so no token is needed. What the graph produces, and
what a player edits by hand, is stored in a SQLite file and queued for
upload; `upload` later pushes the queue with a token that may write, and
holds back every unit Weblate has changed since it was read.
"""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from loguru import logger

from src.core.placeholders import validate_tags
from src.models.weblate import (
    CorpusUnitSchema,
    WeblateUnitPatchSchema,
    WeblateUnitSchema,
)
from src.models.workshop import AssetWorkSchema
from src.services.stores.base import Candidates, Prepared, UnitWrite
from src.services.stores.upstream import NeedsEditing, true_sources
from src.services.weblate import WEBLATE_STATE_TRANSLATED, AsyncWeblateClient

type Origin = Literal["machine", "manual"]
type UploadState = Literal["pending", "uploaded", "conflict"]

SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL,
    language TEXT NOT NULL,
    context TEXT NOT NULL,
    UNIQUE (slug, language, context)
);
CREATE TABLE IF NOT EXISTS components (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    slug TEXT NOT NULL,
    context TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT NOT NULL,
    PRIMARY KEY (slug, context)
);
CREATE TABLE IF NOT EXISTS translations (
    slug TEXT NOT NULL,
    language TEXT NOT NULL,
    context TEXT NOT NULL,
    origin TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    state INTEGER NOT NULL,
    comment TEXT NOT NULL,
    upstream_at_read TEXT NOT NULL,
    upload TEXT NOT NULL,
    upstream_now TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (slug, language, context, origin)
);
"""


@dataclass(frozen=True)
class QueuedTranslation:
    """One translation of this machine, as the upload queue shows it."""

    slug: str
    language: str
    context: str
    origin: Origin
    source: str
    target: str
    state: int
    comment: str
    upload: UploadState
    # What Weblate held when this was made, and, for a conflict, now.
    upstream_at_read: str
    upstream_now: str


@dataclass(frozen=True)
class UploadReport:
    uploaded: int
    conflicts: int


@dataclass(frozen=True)
class _Row:
    source: str
    target: str
    state: int


@dataclass
class _Component:
    work: AssetWorkSchema
    # What Weblate holds for each unit, usable or not, as it was read.
    upstream: dict[str, str]
    # Weblate's translations made for the current source, with valid tags.
    current: dict[str, str]
    pending: list[str]


class LocalStore:
    def __init__(self, client: AsyncWeblateClient, path: Path) -> None:
        self._client = client
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)
        self._needs_editing = NeedsEditing(client)
        self._components: dict[tuple[str, str], _Component] = {}

    async def prepare(
        self, work: AssetWorkSchema, *, name: str, language: str
    ) -> Prepared:
        slug = work.asset.component_slug
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO components VALUES (?, ?)", (slug, name)
            )
            self._db.executemany(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?)",
                [(slug, u.context, u.source, u.note) for u in work.units],
            )
        snapshot = await self._read_upstream(work, language)
        upstream = {unit.context: unit.target for unit in snapshot if unit.target}
        current: dict[str, str] = {}
        previous: dict[str, str] = {}
        for unit in snapshot:
            expected = work.expected.get(unit.context)
            if expected is None or not unit.target.strip():
                continue
            if unit.source != expected:
                previous[unit.context] = unit.target
            elif validate_tags(unit.source, unit.target)[0]:
                current[unit.context] = unit.target
        local = self._local(slug, language)
        for (context, _origin), row in local.items():
            if row.source != work.expected.get(context, row.source):
                previous.setdefault(context, row.target)
        held = set(current) | {
            context
            for (context, _origin), row in local.items()
            if row.source == work.expected.get(context)
        }
        pending = [context for context in work.expected if context not in held]
        self._components[slug, language] = _Component(
            work=work, upstream=upstream, current=current, pending=pending
        )
        return Prepared(
            snapshot=[
                unit.model_copy(update={"target": current.get(unit.context, "")})
                for unit in work.units
            ],
            previous={c: t for c, t in previous.items() if c in pending},
            needs_translation=bool(pending),
        )

    async def pending(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        component = self._components[slug, language]
        notes = {unit.context: unit.note for unit in component.work.units}
        return [
            WeblateUnitSchema(
                id=self._unit_id(slug, language, context),
                language_code=language,
                source=component.work.expected[context],
                target="",
                context=context,
                note=notes.get(context, ""),
            )
            for context in component.pending
        ]

    async def write(self, item: UnitWrite) -> None:
        slug, language, context = self._db.execute(
            "SELECT slug, language, context FROM units WHERE id = ?", (item.unit_id,)
        ).fetchone()
        component = self._components[slug, language]
        self._save(
            slug,
            language,
            context,
            origin="machine",
            source=component.work.expected[context],
            target=item.target,
            state=item.state,
            comment=item.comment,
            upstream_at_read=component.upstream.get(context, ""),
        )

    def edit(
        self, slug: str, language: str, context: str, *, source: str, target: str
    ) -> None:
        """Keep a player's own translation; it outranks every other source."""
        component = self._components.get((slug, language))
        upstream = component.upstream.get(context, "") if component else ""
        self._save(
            slug,
            language,
            context,
            origin="manual",
            source=source,
            target=target,
            state=WEBLATE_STATE_TRANSLATED,
            comment="",
            upstream_at_read=upstream,
        )

    async def candidates(self, work: AssetWorkSchema, *, language: str) -> Candidates:
        slug = work.asset.component_slug
        component = self._components[slug, language]
        result: Candidates = {context: {} for context in work.expected}
        for context, target in component.current.items():
            fuzzy = await self._needs_editing.contains(language, slug, context)
            result[context]["needs_editing" if fuzzy else "upstream"] = target
        for (context, origin), row in self._local(slug, language).items():
            if context not in result or row.source != work.expected[context]:
                continue
            if origin == "manual":
                result[context]["manual"] = row.target
            elif row.state >= WEBLATE_STATE_TRANSLATED:
                result[context]["machine"] = row.target
            else:
                result[context].setdefault("needs_editing", row.target)
        return result

    def queue(
        self, *, uploads: tuple[UploadState, ...] = ("pending", "conflict")
    ) -> list[QueuedTranslation]:
        marks = ",".join("?" * len(uploads))
        rows = self._db.execute(
            "SELECT slug, language, context, origin, source, target, state, "
            "comment, upload, upstream_at_read, upstream_now FROM translations "
            f"WHERE upload IN ({marks}) ORDER BY slug, context",
            uploads,
        ).fetchall()
        return [QueuedTranslation(*row) for row in rows]

    async def upload(
        self,
        client: AsyncWeblateClient,
        *,
        language: str,
        slugs: set[str] | None = None,
    ) -> UploadReport:
        """Push the queued translations of `language` with a writing client.

        A component Weblate lacks is created from the stored sources first.
        A unit whose Weblate translation differs from the one read when the
        local translation was made is not overwritten but marked a conflict.
        """
        queued = [
            row
            for row in self.queue(uploads=("pending",))
            if row.language == language and (slugs is None or row.slug in slugs)
        ]
        uploaded = conflicts = 0
        for slug in sorted({row.slug for row in queued}):
            await client.sync_corpus(
                slug,
                name=self._component_name(slug),
                language=language,
                units=self._sources(slug),
            )
            held = {u.context: u for u in await client.list_units(slug, language)}
            for row in (r for r in queued if r.slug == slug):
                unit = held.get(row.context)
                if unit is None or unit.source != row.source:
                    continue
                if unit.target != row.upstream_at_read:
                    self._mark(row, "conflict", upstream_now=unit.target)
                    conflicts += 1
                    continue
                await client.patch_unit(
                    unit.id,
                    WeblateUnitPatchSchema(target=[row.target], state=row.state),
                )
                if row.comment:
                    await client.add_unit_comment(unit.id, row.comment)
                self._mark(row, "uploaded")
                uploaded += 1
        logger.info("Uploaded {}; {} conflicts held back", uploaded, conflicts)
        return UploadReport(uploaded=uploaded, conflicts=conflicts)

    def resolve(
        self,
        slug: str,
        language: str,
        context: str,
        origin: Origin,
        *,
        keep: Literal["local", "upstream"],
    ) -> None:
        """Settle a conflict: queue the local one again, or drop it."""
        key = (slug, language, context, origin)
        with self._db:
            if keep == "upstream":
                self._db.execute(
                    "DELETE FROM translations WHERE slug = ? AND language = ? "
                    "AND context = ? AND origin = ?",
                    key,
                )
            else:
                self._db.execute(
                    "UPDATE translations SET upload = 'pending', "
                    "upstream_at_read = upstream_now, upstream_now = '' "
                    "WHERE slug = ? AND language = ? AND context = ? AND origin = ?",
                    key,
                )

    async def aclose(self) -> None:
        await self._needs_editing.aclose()
        self._db.close()

    async def _read_upstream(
        self, work: AssetWorkSchema, language: str
    ) -> list[CorpusUnitSchema]:
        slug = work.asset.component_slug
        translations = await self._client.component_translations(slug)
        held = translations.get(language) if translations else None
        if held is None or not held.total:
            return []
        return await true_sources(
            self._client, work, await self._client.download_units(slug, language)
        )

    def _local(self, slug: str, language: str) -> dict[tuple[str, str], _Row]:
        rows = self._db.execute(
            "SELECT context, origin, source, target, state FROM translations "
            "WHERE slug = ? AND language = ?",
            (slug, language),
        ).fetchall()
        return {(r[0], r[1]): _Row(source=r[2], target=r[3], state=r[4]) for r in rows}

    def _unit_id(self, slug: str, language: str, context: str) -> int:
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO units (slug, language, context) VALUES (?, ?, ?)",
                (slug, language, context),
            )
        return self._db.execute(
            "SELECT id FROM units WHERE slug = ? AND language = ? AND context = ?",
            (slug, language, context),
        ).fetchone()[0]

    def _save(
        self,
        slug: str,
        language: str,
        context: str,
        *,
        origin: Origin,
        source: str,
        target: str,
        state: int,
        comment: str,
        upstream_at_read: str,
    ) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO translations VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', ?)",
                (
                    slug,
                    language,
                    context,
                    origin,
                    source,
                    target,
                    state,
                    comment,
                    upstream_at_read,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def _mark(
        self, row: QueuedTranslation, upload: UploadState, *, upstream_now: str = ""
    ) -> None:
        with self._db:
            self._db.execute(
                "UPDATE translations SET upload = ?, upstream_now = ? "
                "WHERE slug = ? AND language = ? AND context = ? AND origin = ?",
                (upload, upstream_now, row.slug, row.language, row.context, row.origin),
            )

    def _component_name(self, slug: str) -> str:
        return self._db.execute(
            "SELECT name FROM components WHERE slug = ?", (slug,)
        ).fetchone()[0]

    def _sources(self, slug: str) -> list[CorpusUnitSchema]:
        rows = self._db.execute(
            "SELECT context, source, note FROM sources WHERE slug = ?", (slug,)
        ).fetchall()
        return [
            CorpusUnitSchema(context=c, source=s, target="", note=n) for c, s, n in rows
        ]
