"""The local app's own settings, edited in its pages and kept per user.

Secrets are not part of them: they live in the OS credential store (see
`src.app.secrets`) and are joined in only when a pipeline config is built.
"""

import os
import secrets
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr, field_validator

from src.config import (
    AgentDefaultsSchema,
    LocalConfigSchema,
    ServiceConfigSchema,
)
from src.models._share import (
    DEFAULT_LLM_CONCURRENCY,
    MAX_LLM_CONCURRENCY,
    BaseSchema,
)
from src.models.store import DEFAULT_PRIORITY, SourceKind, StoreMode, check_priority
from src.models.weblate import WeblateConfigSchema
from src.models.workshop import SteamConfigSchema

SETTINGS_FILE: Final = "settings.json"

type UiLanguage = Literal["zh-CN", "en"]


def app_home() -> Path:
    """Where the app keeps its settings, translation store and caches.

    `X2LOC_HOME` overrides the per-user default, e.g. for tests.
    """
    override = os.environ.get("X2LOC_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share"
    return Path(base) / "x2loc"


class ModelSettingsSchema(BaseSchema):
    base_url: str = "https://openrouter.ai/api/v1"
    translation_model: str = ""
    # Quality-gate retries; empty uses `translation_model`.
    retry_translation_model: str = ""
    validation_model: str = ""
    scoring_model: str = ""
    # The values the project's own runs use.
    batch_size: int = Field(default=200, ge=1)
    units_per_request: int = Field(default=5, ge=1)
    auto_approve_threshold: int = Field(default=90, ge=0, le=100)
    timeout_seconds: float = Field(default=180, gt=0)
    concurrency: int = Field(
        default=DEFAULT_LLM_CONCURRENCY, ge=1, le=MAX_LLM_CONCURRENCY
    )


class WeblateSettingsSchema(BaseSchema):
    url: str = "https://weblate.chat-yu.net/api"
    project_slug: str = "x2loc"


class PathSettingsSchema(BaseSchema):
    workshop_dir: Path | None = None
    mods_dir: Path | None = None


class AppSettingsSchema(BaseSchema):
    model: ModelSettingsSchema = Field(default_factory=ModelSettingsSchema)
    weblate: WeblateSettingsSchema = Field(default_factory=WeblateSettingsSchema)
    paths: PathSettingsSchema = Field(default_factory=PathSettingsSchema)
    store: StoreMode = StoreMode.AUTO
    priority: list[SourceKind] = Field(default_factory=lambda: list(DEFAULT_PRIORITY))
    # Mods translated at once; the LLM ceiling is shared across them.
    jobs: int = Field(default=4, ge=1)
    language: UiLanguage = "zh-CN"

    @field_validator("priority")
    @classmethod
    def _check_priority(cls, priority: list[SourceKind]) -> list[SourceKind]:
        return check_priority(priority)

    def to_service_config(
        self, *, home: Path, llm_api_key: str, weblate_token: str
    ) -> ServiceConfigSchema:
        """The pipeline config these settings and secrets describe."""
        paths = self.paths
        return ServiceConfigSchema.from_values(
            # Only the HTTP service checks it; the app has its own session.
            service_token=SecretStr(secrets.token_urlsafe(16)),
            data_root=home / "work",
            glossary_cache_dir=home / "glossaries",
            store_path=home / "store.sqlite",
            store=self.store,
            priority=self.priority,
            steam=SteamConfigSchema(steam_username="", steam_password=SecretStr("")),
            weblate=WeblateConfigSchema(
                url=self.weblate.url,
                token=SecretStr(weblate_token),
                project_slug=self.weblate.project_slug,
            ),
            agent=AgentDefaultsSchema(
                api_key=SecretStr(llm_api_key),
                base_url=self.model.base_url,
                translation_model_name=self.model.translation_model,
                retry_translation_model_name=self.model.retry_translation_model,
                validate_model_name=self.model.validation_model,
                scoring_model_name=self.model.scoring_model,
                batch_size=self.model.batch_size,
                units_per_request=self.model.units_per_request,
                auto_approve_threshold=self.model.auto_approve_threshold,
                llm_timeout_seconds=self.model.timeout_seconds,
            ),
            local=(
                LocalConfigSchema(
                    workshop_dir=paths.workshop_dir, mods_dir=paths.mods_dir
                )
                if paths.workshop_dir and paths.mods_dir
                else None
            ),
        )


def load_settings(home: Path) -> AppSettingsSchema:
    path = home / SETTINGS_FILE
    if not path.is_file():
        return AppSettingsSchema()
    return AppSettingsSchema.model_validate_json(path.read_text(encoding="utf-8"))


def save_settings(home: Path, settings: AppSettingsSchema) -> None:
    """Replace the settings file whole, so a crash never leaves half of it."""
    home.mkdir(parents=True, exist_ok=True)
    staging = home / f"{SETTINGS_FILE}.tmp"
    staging.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    staging.replace(home / SETTINGS_FILE)
