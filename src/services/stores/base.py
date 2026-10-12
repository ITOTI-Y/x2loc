"""Where a job reads translations from and writes them to.

The translation graph needs two operations, `pending` and `write`
(`UnitStore`); the pipeline adds `prepare` before translating and
`candidates` after it (`TranslationStore`). Each store reports, for every
unit, the translation it holds from each kind of source; `merge` then picks
one per unit by the configured priority, the same way for every store.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from src.models.store import SourceKind
from src.models.weblate import CorpusUnitSchema, WeblateUnitSchema
from src.models.workshop import AssetWorkSchema

# unit context -> source kind -> translation
type Candidates = dict[str, dict[SourceKind, str]]


class UnitWrite(NamedTuple):
    unit_id: int
    target: str
    state: int
    # Empty for none; otherwise posted on the unit after the target.
    comment: str


@dataclass(frozen=True)
class Prepared:
    """A component ready for translation."""

    # The component as it stands, for the cross-component context index.
    snapshot: list[CorpusUnitSchema]
    # Translations made for a source the mod has since changed, by context:
    # each is scored as its unit's first-round candidate, not trusted.
    previous: dict[str, str]
    needs_translation: bool


class UnitStore(Protocol):
    async def pending(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        """Units of the component that hold no translation yet."""
        ...

    async def write(self, item: UnitWrite) -> None:
        """Store one translation decision; raises if it cannot."""
        ...


class TranslationStore(UnitStore, Protocol):
    async def prepare(
        self, work: AssetWorkSchema, *, name: str, language: str
    ) -> Prepared: ...

    async def candidates(self, work: AssetWorkSchema, *, language: str) -> Candidates:
        """Every source's current translation of each expected unit."""
        ...

    async def aclose(self) -> None: ...


def merge(candidates: Candidates, priority: Iterable[SourceKind]) -> dict[str, str]:
    """One translation per unit: the first kind in `priority` it has."""
    order = list(priority)
    merged: dict[str, str] = {}
    for context, by_kind in candidates.items():
        for kind in order:
            if by_kind.get(kind):
                merged[context] = by_kind[kind]
                break
    return merged
