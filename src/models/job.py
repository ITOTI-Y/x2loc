from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, field_validator

from src.agent._share import DEFAULT_LLM_CONCURRENCY, MAX_LLM_CONCURRENCY
from src.core.workshop import WorkshopInputError, parse_workshop_url
from src.models._share import BaseSchema
from src.models.workshop import TARGET_LANGUAGE


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobStage(StrEnum):
    DOWNLOADING = "downloading"
    DISCOVERING = "discovering"
    SYNCING_WEBLATE = "syncing_weblate"
    TRANSLATING = "translating"
    WRITING = "writing"
    EXTRACTING_TERMS = "extracting_terms"
    PACKAGING = "packaging"


ACTIVE_STATUSES = frozenset({JobStatus.QUEUED, JobStatus.RUNNING})
TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.CANCELLED, JobStatus.FAILED}
)


class WorkshopJobRequestSchema(BaseSchema):
    workshop_url: HttpUrl
    target_lang: str = TARGET_LANGUAGE
    llm_api_base_url: HttpUrl
    llm_api_key: SecretStr = Field(repr=False)
    translation_model: str = Field(min_length=1)
    validation_model: str = ""
    scoring_model: str = ""
    llm_concurrency: int = Field(
        default=DEFAULT_LLM_CONCURRENCY, ge=1, le=MAX_LLM_CONCURRENCY
    )

    @field_validator("target_lang")
    @classmethod
    def validate_target_lang(cls, value: str) -> str:
        if value != TARGET_LANGUAGE:
            raise ValueError(f"only {TARGET_LANGUAGE} is supported")
        return value

    @field_validator("llm_api_base_url")
    @classmethod
    def validate_llm_url(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("LLM base URL must use HTTPS")
        return value

    @field_validator("workshop_url")
    @classmethod
    def validate_workshop_url(cls, value: HttpUrl) -> HttpUrl:
        """Reject malformed Workshop URLs while still at the request boundary.

        Doing this here turns a bad URL into a 422 instead of a job that
        fails in its first stage.
        """
        try:
            parse_workshop_url(str(value))
        except WorkshopInputError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @property
    def workshop_id(self) -> str:
        return parse_workshop_url(str(self.workshop_url))


class JobProgressSchema(BaseSchema):
    files_total: int = 0
    files_completed: int = 0
    units_total: int = 0
    units_translated: int = 0
    terms_added: int = 0
    terms_skipped: int = 0


class ArtifactSchema(BaseSchema):
    path: Path
    sha256: str
    bytes: int


class JobRecordSchema(BaseSchema):
    id: str
    workshop_id: str
    workshop_url: str
    title: str = ""
    target_lang: str
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage | None = None
    progress: JobProgressSchema = Field(default_factory=JobProgressSchema)
    error_code: str | None = None
    artifact: ArtifactSchema | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


class JobUpdateSchema(BaseSchema):
    """Partial update for one job record.

    Only explicitly assigned fields are applied — `stage=None` clears the
    stage, while an untouched field leaves the record value as-is. The
    distinction comes from `model_fields_set`, so construct this schema with
    keyword arguments only for the fields being changed.
    """

    status: JobStatus | None = None
    stage: JobStage | None = None
    progress: JobProgressSchema | None = None
    error_code: str | None = None
    artifact: ArtifactSchema | None = None


class JobCreateResponseSchema(BaseSchema):
    id: str
    status: JobStatus
    status_url: str
    events_url: str


class JobStatusResponseSchema(BaseSchema):
    id: str
    workshop_id: str
    title: str
    target_lang: str
    status: JobStatus
    stage: JobStage | None
    progress: JobProgressSchema
    error_code: str | None
    artifact_sha256: str | None
    artifact_bytes: int | None
    artifact_url: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None
