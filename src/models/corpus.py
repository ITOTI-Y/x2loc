from pathlib import Path

from pydantic import Field, computed_field

from src.models._share import BaseSchema
from src.models.entry import EntrySchema
from src.models.mod import BASE_GAME_NAMESPACE, BASE_GAME_TITLE
from src.models.section import SectionHeader


class BilingualEntry(BaseSchema):
    compound_key: str
    source: EntrySchema
    target: EntrySchema | None = None
    section_header: SectionHeader


class BilingualCorpus(BaseSchema):
    source_lang: str
    target_lang: str
    source_path: Path
    target_path: Path | None = None
    entries: list[BilingualEntry] = Field(default_factory=list)
    source_only: list[str] = Field(default_factory=list)
    target_only: list[str] = Field(default_factory=list)

    # Namespace identity — populated by align-dir from mod_resolver output.
    # `namespace` is the Weblate component-slug prefix (e.g. "1122837889-
    # more-traits", "local-wc-quick-lw2", or "base-xcom2-wotc"), used by
    # upload/download/writeback so they never need to re-read .XComMod.
    # Defaults to the base-game constant when the corpus is produced
    # outside the mod-resolver pipeline (tests, manual construction).
    namespace: str = Field(
        default=BASE_GAME_NAMESPACE,
        description=(
            "Stable mod identity used as the Weblate component-slug prefix "
            "and output subdirectory name."
        ),
    )
    steam_id: str | None = Field(
        default=None,
        description="Steam Workshop ID, if known.",
    )
    mod_title: str = Field(
        default=BASE_GAME_TITLE,
        description="Human-readable mod title from the .XComMod manifest.",
    )

    @computed_field
    @property
    def aligned_count(self) -> int:
        target_only_set = set(self.target_only)
        return sum(
            1
            for e in self.entries
            if e.target is not None and e.compound_key not in target_only_set
        )
