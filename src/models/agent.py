from typing import Literal, TypedDict

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from src.models._share import BaseSchema
from src.models.weblate import WeblateUnitSchema


class PatternExampleSchema(TypedDict):
    source: str
    target: str


class PatternSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore")
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


class DeductionSchema(BaseSchema):
    dim: str = Field(description="The dimension of the deduction")
    pts: int = Field(description="The points of the deduction")
    reason: str = Field(description="The reason for the deduction")


class ScoreResultSchema(BaseSchema):
    score: int = Field(
        ge=0, le=100, description="The score of the translation between 0 and 100"
    )
    deductions: list[DeductionSchema] = Field(
        default_factory=list,
        description="The deductions for the score, if no deduction is needed, return an empty list",
    )
    suggested_translation: str | None = Field(
        None,
        description="The suggested translation if the score is less than the threshold, must reply in target language, if no suggested translation is needed, return null",
    )
    notes: str | None = Field(
        None, description="The notes for the score if not needed, return null"
    )


class TranslationUnitSchema(BaseSchema):
    id: int
    source: str
    translated: str
    key: str
    context: list[ComponentInfoSchema]
    category: str
    pattern_matched: bool
    glossary_base: list[WeblateUnitSchema]
    glossary_mods: list[WeblateUnitSchema]
    tag_valid: bool
    original_unit: WeblateUnitSchema
    patterns: list[PatternSchema]
    suggested_translation: str | None = None
    score_result: ScoreResultSchema | None = None


class ReviewItemSchema(BaseSchema):
    unit_id: int
    source: str
    translation: str
    category: str
    score: int
    deductions: list[DeductionSchema]
    suggested_translation: str | None


class ReviewDecisionSchema(BaseSchema):
    unit_id: int
    action: Literal["approve", "modify", "skip"]
    translation: str | None = None


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

    scores: list[TranslationUnitSchema] = Field(default_factory=list)
