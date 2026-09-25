from pydantic import SecretStr

from src.agent._share import MAX_TRANSLATION_ATTEMPTS
from src.config import AgentDefaultsSchema, ServiceConfigSchema
from src.models._share import BaseSchema
from src.models.weblate import WeblateConfigSchema
from src.models.workshop import SteamConfigSchema


class ConfigSchema(BaseSchema):
    """Fully resolved settings for one agent run.

    Defaults live in `src.config`; build instances with `build_agent_config`.
    """

    weblate: WeblateConfigSchema
    steam: SteamConfigSchema
    translation_model_name: str
    validate_model_name: str
    scoring_model_name: str
    base_url: str
    api_key: SecretStr
    translation_temperature: float = 0.0
    validate_temperature: float = 0.0
    scoring_temperature: float = 0.0
    batch_size: int
    auto_approve_threshold: int
    max_translation_attempts: int = MAX_TRANSLATION_ATTEMPTS
    max_concurrency: int
    base_glossary_slug: str
    mods_glossary_slug: str
    custom_glossary_slug: str
    target_lang: str

    @property
    def effective_validate_model(self) -> str:
        return self.validate_model_name or self.translation_model_name

    @property
    def effective_scoring_model(self) -> str:
        return self.scoring_model_name or self.translation_model_name


def build_agent_config(
    service: ServiceConfigSchema,
    agent: AgentDefaultsSchema,
    *,
    target_lang: str,
    max_concurrency: int,
) -> ConfigSchema:
    """Combine service-wide settings with the (possibly overridden) LLM table."""
    if not agent.api_key.get_secret_value() or not agent.translation_model_name:
        raise ValueError(
            "no LLM api key or translation model: provide them in the "
            "request or in the [agent] table of the service TOML"
        )
    return ConfigSchema(
        weblate=service.weblate,
        steam=service.steam,
        base_glossary_slug=service.glossary.base_slug,
        mods_glossary_slug=service.glossary.mods_slug,
        custom_glossary_slug=service.glossary.custom_slug,
        translation_model_name=agent.translation_model_name,
        validate_model_name=agent.validate_model_name,
        scoring_model_name=agent.scoring_model_name,
        base_url=agent.base_url,
        api_key=agent.api_key,
        batch_size=agent.batch_size,
        auto_approve_threshold=agent.auto_approve_threshold,
        target_lang=target_lang,
        max_concurrency=max_concurrency,
    )
