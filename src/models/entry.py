from pydantic import Field

from src.models._share import BaseSchema, PlaceholderType


class PlaceholderSchema(BaseSchema):
    pattern: str
    type: PlaceholderType
    span: tuple[int, int]  # (start, end) within the value string


class StructFieldSchema(BaseSchema):
    key: str
    raw_value: str  # original value including quotes
    value: str  # value without quotes
    placeholders: list[PlaceholderSchema] = Field(default_factory=list)


class EntrySchema(BaseSchema):
    key: str
    raw_value: str  # original value including quotes
    value: str  # value without quotes
    is_array: bool = False
    array_index: int | None = None
    is_append: bool = False  # +Key= syntax
    struct_fields: list[StructFieldSchema] | None = None
    placeholders: list[PlaceholderSchema] = Field(default_factory=list)
    comments: list[str] = Field(default_factory=list)
    line_number: int = Field(default=0)
