from typing import TypedDict

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from src.models._share import BaseSchema
from src.models.weblate import WeblateUnitSchema


class PatternExampleSchema(TypedDict):
    source: str
    target: str


class PatternSchema(TypedDict):
    src_pattern: str
    tgt_pattern: str
    approved_count: int
    examples: list[PatternExampleSchema]


class StatsSchema(TypedDict):
    auto: int
    approved: int
    modified: int
    skipped: int

@dataclass
class ComponentInfoSchema:
    slug: str
    lang: str
    positions: int
    nearby: list[int] = Field(default_factory=list)
    translated_percent: float = 0.0


class NewAgentStateSchema(BaseSchema):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    stats: StatsSchema = Field(
        default_factory=lambda: StatsSchema(auto=0, approved=0, modified=0, skipped=0)
    )

    base_glossary: dict[str, WeblateUnitSchema] = Field(default_factory=dict)
    mods_glossary: dict[str, WeblateUnitSchema] = Field(default_factory=dict)

    current_page: int = Field(default=1)
    skip_ids: list[int] = Field(default_factory=list)

    should_continue: bool = Field(default=True)
    patterns: dict[str, PatternSchema] = Field(default_factory=dict)

    to_translate: list[WeblateUnitSchema] = Field(default_factory=list)
    is_end: bool = Field(default=False)
