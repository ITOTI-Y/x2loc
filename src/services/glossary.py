import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from src.config import GlossaryConfigSchema
from src.core.glossary import normalize_term, term_context, term_pair
from src.models.glossary import GlossaryTerm
from src.models.weblate import (
    CorpusUnitSchema,
    WeblateUnitDraftSchema,
    WeblateUnitSchema,
)
from src.services.weblate import AsyncWeblateClient, WeblateAPIError


class GlossarySource(Protocol):
    """Where the translation graph reads glossary units from."""

    async def units(self, slug: str, language: str) -> list[WeblateUnitSchema]: ...

    async def aclose(self) -> None: ...


class GlossaryWriter(Protocol):
    """Where a job appends newly mined term pairs; returns (added, skipped)."""

    async def write(self, terms: list[GlossaryTerm]) -> tuple[int, int]: ...


def term_pairs(terms: Iterable[GlossaryTerm]) -> list[tuple[str, str]]:
    """Normalized, deduplicated pairs worth storing (target differs)."""
    return sorted(
        {
            term_pair(term.source, term.target)
            for term in terms
            if normalize_term(term.source)
            and normalize_term(term.target)
            and normalize_term(term.source) != normalize_term(term.target)
        }
    )


def existing_pairs(units: Iterable[WeblateUnitSchema]) -> set[tuple[str, str]]:
    return {
        term_pair(unit.source, unit.target)
        for unit in units
        if normalize_term(unit.source) and normalize_term(unit.target)
    }


async def publish_pairs(
    client: AsyncWeblateClient,
    *,
    component_slug: str,
    target_lang: str,
    pairs: list[tuple[str, str]],
) -> None:
    """Create term pairs in a glossary component; safe to repeat.

    Two-phase by Weblate's template-component contract: source strings are
    created on the source translation (the only place unit creation is
    allowed), then the targets are filled through one translate upload.
    """

    async def _create(source: str, target: str) -> None:
        try:
            await client.create_unit(
                component_slug,
                WeblateUnitDraftSchema(
                    context=term_context(source, target), source=source
                ),
            )
        except WeblateAPIError as exc:
            # Creation is not idempotent on Weblate's side: a retry of a
            # slow-but-successful POST, or a source string left over from an
            # earlier partial run, answers 400 "already exists". The
            # key/value are code-generated, so 400 here cannot mean a
            # validation error; the upload below fills the target either
            # way, which also heals empty-target orphans.
            if exc.status_code != 400:
                raise

    await asyncio.gather(*(_create(source, target) for source, target in pairs))
    await client.upload_targets(
        component_slug,
        target_lang,
        [
            CorpusUnitSchema(
                context=term_context(source, target),
                source=source,
                target=target,
                note="",
            )
            for source, target in pairs
        ],
    )


@dataclass(frozen=True)
class _Snapshot:
    started_at: float
    units: asyncio.Task[list[WeblateUnitSchema]]


class GlossarySnapshots:
    """Translated glossary units shared by every job of one process.

    A full glossary read takes minutes on a slow Weblate, so each glossary
    is fetched once and reused until `ttl_seconds` pass or `invalidate`
    marks it stale; `CustomGlossaryWriter` invalidates its glossary right
    after writing, so the next reader sees the new terms. Reads are
    single-flight and shielded: a cancelled caller never aborts a read that
    others are waiting on, and a failed read is not reused.
    """

    def __init__(self, client: AsyncWeblateClient, *, ttl_seconds: float) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._snapshots: dict[tuple[str, str], _Snapshot] = {}

    async def units(self, slug: str, language: str) -> list[WeblateUnitSchema]:
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

    def invalidate(self, slug: str) -> None:
        self._snapshots = {
            key: snapshot for key, snapshot in self._snapshots.items() if key[0] != slug
        }

    async def aclose(self) -> None:
        """Cancel reads still in flight; the owner calls this before closing
        the Weblate client."""
        pending = [s.units for s in self._snapshots.values() if not s.units.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._snapshots.clear()

    def _reusable(self, snapshot: _Snapshot) -> bool:
        if time.monotonic() - snapshot.started_at > self._ttl_seconds:
            return False
        task = snapshot.units
        return not task.done() or (not task.cancelled() and task.exception() is None)


class CustomGlossaryWriter:
    """Appends newly mined term pairs to the writable custom glossary."""

    def __init__(
        self,
        client: AsyncWeblateClient,
        snapshots: GlossarySnapshots,
        *,
        component_slug: str,
        target_lang: str,
    ) -> None:
        self._client = client
        self._snapshots = snapshots
        self._component_slug = component_slug
        self._target_lang = target_lang
        self._lock = asyncio.Lock()

    async def write(self, terms: list[GlossaryTerm]) -> tuple[int, int]:
        """Create the pairs that are not present yet; return (added, skipped).

        The lock covers only the read-and-diff; creation runs concurrently
        under the client's own semaphore. The diff reads the shared snapshot
        instead of the whole component. A stale snapshot only makes a pair
        look new, and re-creating an existing pair is tolerated; the snapshot
        is invalidated once anything may have been written.
        """
        pairs = term_pairs(terms)
        if not pairs:
            return 0, 0
        async with self._lock:
            current = await self._snapshots.units(
                self._component_slug, self._target_lang
            )
            existing = existing_pairs(current)
            new_pairs = [pair for pair in pairs if pair not in existing]
        if not new_pairs:
            return 0, len(pairs)
        try:
            await publish_pairs(
                self._client,
                component_slug=self._component_slug,
                target_lang=self._target_lang,
                pairs=new_pairs,
            )
        finally:
            self._snapshots.invalidate(self._component_slug)
        return len(new_pairs), len(pairs) - len(new_pairs)


async def validate_weblate_components(
    client: AsyncWeblateClient, glossary: GlossaryConfigSchema
) -> None:
    """Assert the three pre-provisioned glossary components are usable.

    They are operational assets; this implementation never creates or
    migrates them, so a missing one must stop startup rather than surface
    as a confusing mid-job failure.
    """
    for slug in (glossary.base_slug, glossary.mods_slug, glossary.custom_slug):
        component = await client.get_component(slug)
        if component is None:
            raise RuntimeError(f"required Weblate component is missing: {slug}")
        if component.file_format != "csv":
            raise RuntimeError(f"Weblate component must use CSV: {slug}")
        if slug == glossary.custom_slug and not (
            component.manage_units and component.edit_template
        ):
            raise RuntimeError(
                "custom glossary must enable manage units and edit template"
            )
