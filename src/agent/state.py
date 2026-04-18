import operator
from typing import Annotated, Literal, TypedDict


class TranslationUnit(TypedDict):
    id: int
    position: int
    source: str
    context: str
    category: str | None


class ContextResult(TypedDict):
    unit_id: int
    mod_component: str | None
    translated_percent: float | None
    nearby: list[dict]


class GlossaryMatch(TypedDict):
    source: str
    target: str
    context: str


class TranslationCandidate(TypedDict):
    unit_id: int
    source: str
    context: str
    category: str | None
    translation: str
    pattern_matched: bool
    glossary_base: list[GlossaryMatch]
    glossary_mods: list[GlossaryMatch]
    context_result: ContextResult
    tag_valid: bool


class Deduction(TypedDict):
    dim: str
    pts: int
    reason: str


class ScoreResult(TypedDict):
    unit_id: int
    score: int
    deductions: list[Deduction]
    suggested_alternative: str | None
    notes: str | None


class PatchResult(TypedDict):
    unit_id: int
    status: Literal["ok", "tag_fail", "error"]
    error: str | None


class SessionPattern(TypedDict):
    src_pattern: str
    tgt_pattern: str
    approved_count: int
    examples: list[dict[Literal["source", "target"], str]]


class AgentState(TypedDict):
    base_glossary: dict[str, dict]
    mods_glossary: dict[str, dict]

    current_page: int
    remaining_count: int

    current_units: list[TranslationUnit]
    context_results: list[ContextResult]
    candidates: list[TranslationCandidate]
    scores: list[ScoreResult]

    auto_batch: list[TranslationCandidate]
    review_batch: list[TranslationCandidate]
    review_approved: list[dict]

    patch_results: Annotated[list[PatchResult], operator.add]
    approved_history: Annotated[list[dict], operator.add]

    skip_ids: list[int]
    stats: dict[str, int]
    session_patterns: list[SessionPattern]

    should_continue: bool
    needs_review: bool

    target_lang: str
