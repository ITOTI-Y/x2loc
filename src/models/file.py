from pathlib import Path

from pydantic import Field, computed_field

from src.models._share import BaseSchema
from src.models.section import SectionSchema


class LocalizationFile(BaseSchema):
    path: Path
    lang: str
    encoding: str = Field(default="utf-16-le")
    header_comments: list[str] = Field(default_factory=list)
    sections: list[SectionSchema] = Field(default_factory=list)

    @computed_field
    @property
    def entry_count(self) -> int:
        return sum(len(section.entries) for section in self.sections)
