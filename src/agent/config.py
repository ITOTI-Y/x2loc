import tomllib
from pathlib import Path

from pydantic import Field, SecretStr

from src.agent._share import AUTO_APPROVE_THRESHOLD, DEFAULT_BATCH_SIZE
from src.models._share import BaseSchema
from src.models.weblate import WeblateConfigSchema


class AgentConfigSchema(BaseSchema):
    weblate: WeblateConfigSchema
    translation_model_name: str = "claude-sonnet-4-20250514"
    validate_model_name: str = "gemini-3.1-flash-lite-preview"
    scoring_model_name: str = "gemini-3.1-flash-lite-preview"
    base_url: str = "https://api.anthropic.com/v1"
    api_key: SecretStr = Field(...)
    translation_temperature: float = 0.2
    scoring_temperature: float = 0.0
    batch_size: int = DEFAULT_BATCH_SIZE
    auto_approve_threshold: int = AUTO_APPROVE_THRESHOLD
    component_slug: str = "glossary-mods"
    base_glossary_slug: str = "glossary-base-xcom2-wotc"
    target_lang: str = "zh_Hans"
    dry_run: bool = False


def load_config(
    weblate_config_path: str | Path = "configs/weblate.toml",
) -> AgentConfigSchema:
    with open(weblate_config_path, "rb") as f:
        raw = tomllib.load(f)

    if raw.get("validate_model_name") is None and "translation_model_name" in raw:
        raw["validate_model_name"] = raw["translation_model_name"]
    if raw.get("scoring_model_name") is None and "translation_model_name" in raw:
        raw["scoring_model_name"] = raw["translation_model_name"]

    weblate = WeblateConfigSchema(
        **{k: v for k, v in raw.items() if k in WeblateConfigSchema.model_fields}
    )
    agent_fields = {
        k: v
        for k, v in raw.items()
        if k in AgentConfigSchema.model_fields and k != "weblate"
    }
    if "api_key" in agent_fields:
        agent_fields["api_key"] = SecretStr(agent_fields["api_key"])

    return AgentConfigSchema(weblate=weblate, **agent_fields)
