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
    example_count: int
    examples: list[PatternExampleSchema]


class AgentMessageSchema(TypedDict):
    role: Literal["user"]
    content: str


class AgentInputSchema(TypedDict):
    messages: list[AgentMessageSchema]


class StructuredAgentResponseSchema[T](TypedDict, total=False):
    structured_response: T


class StatsSchema(TypedDict):
    auto: int
    approved: int
    modified: int
    skipped: int
    needs_editing: int


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


class TranslationItemSchema(BaseSchema):
    id: int = Field(description="The id from the item's `## Item <id>` header")
    result: str = Field(
        description="The translated result in the target language only; if the source contains no human-readable text, return the source unchanged"
    )


class TranslationBatchOutputSchema(BaseSchema):
    results: list[TranslationItemSchema] = Field(
        description="Exactly one entry per item in the user message"
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
    suggested_translation: str = Field(
        "",
        description='The improved translation in the target language, REQUIRED when score < 95 per the decision rules; return an empty string "" when the translation is production-ready',
    )
    notes: str = Field(
        "",
        description='The notes for the score in the target language; return an empty string "" if not needed',
    )


class ScoreItemSchema(ScoreResultSchema):
    id: int = Field(description="The id from the item's `## Item <id>` header")


class ScoreBatchOutputSchema(BaseSchema):
    results: list[ScoreItemSchema] = Field(
        description="Exactly one entry per item in the user message"
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
    suggested_translation: str = ""
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
    action: Literal["approve", "modify", "skip", "needs_editing"]
    translation: str | None = None


class NewAgentStateSchema(BaseSchema):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    component_slug: str = ""

    stats: StatsSchema = Field(
        default_factory=lambda: StatsSchema(
            auto=0, approved=0, modified=0, skipped=0, needs_editing=0
        )
    )

    base_glossary: dict[str, tuple[WeblateUnitSchema, ...]] = Field(
        default_factory=dict
    )
    mods_glossary: dict[str, tuple[WeblateUnitSchema, ...]] = Field(
        default_factory=dict
    )

    current_page: int = Field(default=1)
    skip_ids: list[int] = Field(default_factory=list)

    should_continue: bool = Field(default=True)
    patterns: dict[str, tuple[PatternSchema, ...]] = Field(default_factory=dict)
    glossaries_loaded: bool = Field(default=False)

    to_translate: list[WeblateUnitSchema] = Field(default_factory=list)
    is_end: bool = Field(default=False)

    context_results: dict[int, list[ComponentInfoSchema]] = Field(default_factory=dict)

    candidates: list[TranslationUnitSchema] = Field(default_factory=list)

    scores: list[TranslationUnitSchema] = Field(default_factory=list)

    decisions: list[ReviewDecisionSchema] = Field(default_factory=list)
    accepted_decisions: list[ReviewDecisionSchema] = Field(default_factory=list)

    quality_feedback: dict[int, str] = Field(default_factory=dict)
    attempts: int = Field(default=0)
    # Highest-scoring tag-valid candidate per unit across the quality-gate
    # rounds of the current batch: (score, translation).
    best_candidates: dict[int, tuple[int, str]] = Field(default_factory=dict)
    retry_pending: bool = Field(default=False)

    approved_pairs: dict[str, str] = Field(default_factory=dict)
