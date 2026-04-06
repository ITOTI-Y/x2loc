from pydantic import Field

from src.models._share import BaseSchema, SectionHeaderFormat
from src.models.entry import EntrySchema


class SectionHeader(BaseSchema):
    raw: str
    format: SectionHeaderFormat
    name: str
    object_name: str | None = None
    class_name: str | None = None
    package: str | None = None


class SectionSchema(BaseSchema):
    header: SectionHeader
    entries: list[EntrySchema] = Field(default_factory=list)
    comments: list[str] = Field(default_factory=list)
