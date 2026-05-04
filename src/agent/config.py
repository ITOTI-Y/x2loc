import tomllib
from pathlib import Path

from pydantic import Field, SecretStr

from src.agent._share import AUTO_APPROVE_THRESHOLD, DEFAULT_BATCH_SIZE
from src.models._share import BaseSchema
from src.models.weblate import WeblateConfigSchema


class AgentConfigSchema(BaseSchema):
    weblate: WeblateConfigSchema
    translation_model_name: str = "gemini-3.1-flash-lite-preview"
    validate_model_name: str = ""
    scoring_model_name: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = Field(...)
    translation_temperature: float = 1.0
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
    weblate_config = raw.get("weblate")
    if weblate_config is None:
        raise ValueError("weblate config not found")
    weblate = WeblateConfigSchema.model_validate(weblate_config)

    agent_config = raw.get("agent")
    if agent_config is None:
        raise ValueError("agent config not found")
    agent_config.update({"weblate": weblate})
    agent_config = AgentConfigSchema.model_validate(agent_config)

    return agent_config
