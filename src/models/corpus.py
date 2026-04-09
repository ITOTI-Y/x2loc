from pathlib import Path

from pydantic import Field, computed_field

from src.models._share import BaseSchema
from src.models.entry import EntrySchema
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

    @computed_field
    @property
    def aligned_count(self) -> int:
        target_only_set = set(self.target_only)
        return sum(
            1
            for e in self.entries
            if e.target is not None and e.compound_key not in target_only_set
        )
