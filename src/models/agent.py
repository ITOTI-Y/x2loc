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
    unit: WeblateUnitSchema
    key: str
    slug: str
    lang: str
    position: int
    nearby: list[WeblateUnitSchema] = Field(default_factory=list)


class SystemBlockSchema(TypedDict):
    type: str
    text: str
    cache_control: dict[str, str]


class TranslationOutputSchema(BaseSchema):
    result: str = Field(
        description='Only output the translated result, must reply in target language, if no suggested translation is needed, return ""'
    )


class TranslationUnitSchema(BaseSchema):
    id: int
    source: str
    translated: str
    context: str
    category: str
    pattern_matched: bool
    glossary_base: list[WeblateUnitSchema]
    glossary_mods: list[WeblateUnitSchema]
    tag_valid: bool
    original_unit: WeblateUnitSchema


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

    context_results: dict[int, list[ComponentInfoSchema]] = Field(default_factory=dict)

    candidates: list[TranslationUnitSchema] = Field(default_factory=list)
