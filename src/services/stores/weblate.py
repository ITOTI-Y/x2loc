"""Translations kept in Weblate, for a token that may write to the project."""

from loguru import logger

from src.core.artifact import ArtifactValidationError
from src.core.placeholders import validate_tags
from src.models.weblate import (
    CorpusUnitSchema,
    WeblateUnitPatchSchema,
    WeblateUnitSchema,
)
from src.models.workshop import AssetWorkSchema
from src.services.stores.base import Candidates, Prepared, UnitWrite
from src.services.stores.upstream import NeedsEditing, true_sources
from src.services.weblate import (
    WEBLATE_STATE_EMPTY,
    WEBLATE_STATE_FUZZY,
    WEBLATE_STATE_TRANSLATED,
    AsyncWeblateClient,
)


class WeblateStore:
    def __init__(self, client: AsyncWeblateClient) -> None:
        self._client = client
        self._needs_editing = NeedsEditing(client)
        # Units handed out by `pending`: id -> (language, slug, context).
        self._handed_out: dict[int, tuple[str, str, str]] = {}
        # Snapshots of components left as they were read, by (slug,
        # language): nothing translates them, so `candidates` needs no
        # second download.
        self._untouched: dict[tuple[str, str], list[CorpusUnitSchema]] = {}

    async def prepare(
        self, work: AssetWorkSchema, *, name: str, language: str
    ) -> Prepared:
        """Sync the component's sources, then repair what Weblate holds.

        Translations of a source the mod has changed are cleared and
        returned as `previous`; translations whose tags disagree with their
        source are cleared to be redone.
        """
        slug = work.asset.component_slug
        snapshot = await self._client.sync_corpus(
            slug, name=name, language=language, units=work.units
        )
        if snapshot is None:
            snapshot = await self._client.download_units(slug, language)
        snapshot = await true_sources(self._client, work, snapshot)
        previous = await self._clear_changed_sources(work, language, snapshot)
        cleared = await self._clear_tag_mismatches(work, language, snapshot)
        needs_translation = (
            bool(previous) or cleared or any(not unit.target for unit in snapshot)
        )
        if not needs_translation:
            self._untouched[slug, language] = snapshot
        return Prepared(
            snapshot=snapshot,
            previous=previous,
            needs_translation=needs_translation,
        )

    async def pending(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        units = await self._client.list_units(slug, language, q="state:empty")
        for unit in units:
            self._handed_out[unit.id] = (language, slug, unit.context)
        return units

    async def write(self, item: UnitWrite) -> None:
        await self._client.patch_unit(
            item.unit_id, WeblateUnitPatchSchema(target=[item.target], state=item.state)
        )
        # A unit handed to a human without its reason is a failed write.
        if item.comment:
            await self._client.add_unit_comment(item.unit_id, item.comment)
        where = self._handed_out.get(item.unit_id)
        if where is not None:
            self._needs_editing.record(*where, fuzzy=item.state == WEBLATE_STATE_FUZZY)

    async def candidates(self, work: AssetWorkSchema, *, language: str) -> Candidates:
        """Weblate's own view of the component.

        Every unit Weblate holds a translation of is either `upstream` or,
        held as "needs editing", `needs_editing`. A component translated in
        this run is read afresh.
        """
        slug = work.asset.component_slug
        units = self._untouched.pop((slug, language), None)
        if units is None:
            units = await true_sources(
                self._client, work, await self._client.download_units(slug, language)
            )
        result: Candidates = {}
        for unit in units:
            expected_source = work.expected.get(unit.context)
            if expected_source is None:
                continue
            if unit.context in result:
                raise ArtifactValidationError(
                    f"{slug} returned a duplicate unit context"
                )
            if unit.source != expected_source:
                raise ArtifactValidationError(f"{slug} returned a stale source unit")
            result[unit.context] = {}
            if unit.target.strip():
                fuzzy = await self._needs_editing.contains(language, slug, unit.context)
                result[unit.context] = {
                    "needs_editing" if fuzzy else "upstream": unit.target
                }
        return result

    async def aclose(self) -> None:
        await self._needs_editing.aclose()

    async def _clear_changed_sources(
        self,
        work: AssetWorkSchema,
        language: str,
        snapshot: list[CorpusUnitSchema],
    ) -> dict[str, str]:
        """Update the sources the mod has changed; clear their translations.

        A held translation was made for the old source, so it is emptied for
        the graph to redo and returned by context, to be scored as the
        unit's first-round candidate instead of trusted.
        """
        changed = {
            unit.context: unit
            for unit in snapshot
            if unit.context in work.expected
            and unit.source != work.expected[unit.context]
        }
        if not changed:
            return {}
        slug = work.asset.component_slug
        # Patching needs unit ids, which only the (slow) units API carries.
        for unit in await self._client.list_units(
            slug, self._client.config.source_language
        ):
            if unit.context in changed:
                await self._client.patch_unit(
                    unit.id,
                    WeblateUnitPatchSchema(
                        target=[work.expected[unit.context]],
                        state=WEBLATE_STATE_TRANSLATED,
                    ),
                )
        for unit in await self._client.list_units(slug, language):
            if unit.context in changed:
                await self._client.patch_unit(
                    unit.id,
                    WeblateUnitPatchSchema(target=[""], state=WEBLATE_STATE_EMPTY),
                )
        logger.warning(
            "Updated {} changed sources in {}; their translations are redone",
            len(changed),
            slug,
        )
        return {
            context: unit.target
            for context, unit in changed.items()
            if unit.target.strip()
        }

    async def _clear_tag_mismatches(
        self,
        work: AssetWorkSchema,
        language: str,
        snapshot: list[CorpusUnitSchema],
    ) -> bool:
        """Empty held translations whose tags disagree with their source.

        Such a translation would fail the overlay check; emptied, it is
        retranslated like any untranslated unit. Returns whether anything
        was cleared.
        """
        slug = work.asset.component_slug
        broken_contexts = {
            unit.context
            for unit in snapshot
            if unit.context in work.expected
            and unit.target.strip()
            and not validate_tags(unit.source, unit.target)[0]
        }
        if not broken_contexts:
            return False
        # Patching needs unit ids, which only the (slow) units API carries.
        broken = [
            unit
            for unit in await self._client.list_units(slug, language)
            if unit.context in broken_contexts
        ]
        for unit in broken:
            await self._client.patch_unit(
                unit.id, WeblateUnitPatchSchema(target=[""], state=WEBLATE_STATE_EMPTY)
            )
        logger.warning(
            "Cleared {} translations with mismatched tags in {}", len(broken), slug
        )
        return True
