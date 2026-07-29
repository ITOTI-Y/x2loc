import tomllib
from pathlib import Path

from pydantic import Field, SecretStr

from src.agent._share import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LLM_CONCURRENCY,
    MAX_TRANSLATION_ATTEMPTS,
)
from src.models._share import BaseSchema
from src.models.weblate import WeblateConfigSchema
from src.models.workshop import SteamConfigSchema


class ConfigSchema(BaseSchema):
    weblate: WeblateConfigSchema
    steam: SteamConfigSchema
    translation_model_name: str = "gemini-3.1-flash-lite-preview"
    validate_model_name: str = ""
    scoring_model_name: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = Field(...)
    translation_temperature: float = 0.0
    validate_temperature: float = 0.0
    scoring_temperature: float = 0.0
    batch_size: int = DEFAULT_BATCH_SIZE
    auto_approve_threshold: int = 95
    max_translation_attempts: int = MAX_TRANSLATION_ATTEMPTS
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY
    base_glossary_slug: str = "glossary-base-xcom2-wotc"
    mods_glossary_slug: str = "glossary-mods"
    custom_glossary_slug: str = "glossary-custom"
    target_lang: str = "zh_Hans"

    @property
    def effective_validate_model(self) -> str:
        return self.validate_model_name or self.translation_model_name

    @property
    def effective_scoring_model(self) -> str:
        return self.scoring_model_name or self.translation_model_name


def load_config(
    weblate_config_path: str | Path = "configs/weblate.local.toml",
) -> ConfigSchema:
    with open(weblate_config_path, "rb") as f:
        raw = tomllib.load(f)
    weblate = WeblateConfigSchema.model_validate(raw["weblate"])
    steam = SteamConfigSchema.model_validate(raw["steam"])
    return ConfigSchema.model_validate(
        raw["agent"] | {"weblate": weblate, "steam": steam}
    )
