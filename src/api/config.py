from __future__ import annotations

import ipaddress
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from src.models._share import BaseSchema
from src.models.weblate import WeblateConfigSchema
from src.models.workshop import SteamConfigSchema, WorkshopLimitsSchema


class GlossaryConfigSchema(BaseSchema):
    base_slug: str = "glossary-base-xcom2-wotc"
    mods_slug: str = "glossary-mods"
    custom_slug: str = "glossary-custom"


def _default_limits() -> WorkshopLimitsSchema:
    return WorkshopLimitsSchema(
        download_timeout_seconds=1800,
        terminate_grace_seconds=10,
        max_total_bytes=4_000_000_000,
        max_file_count=50_000,
        max_loc_file_bytes=50_000_000,
    )


class ServiceConfigSchema(BaseSettings):
    """Layered service configuration: env > TOML > defaults.

    The TOML source is `configs/weblate.local.toml` — the same file the
    interactive CLI reads, so one gitignored file holds every credential.
    Top-level keys (`service_token`, `data_root`, `bind_port`) sit above
    the first table; `[weblate]`, `[steam]` and `[glossary]` map to the
    nested models. `X2LOC_`-prefixed environment variables override any
    of it for container or CI deployments, nested via `__`, e.g.
    `X2LOC_WEBLATE__TOKEN` and `X2LOC_STEAM__STEAM_PASSWORD`. The path is
    relative to the working directory: run the service from the repository
    root or override via environment.
    """

    model_config = SettingsConfigDict(
        env_prefix="X2LOC_",
        env_nested_delimiter="__",
        toml_file="configs/weblate.local.toml",
        extra="ignore",
    )

    service_token: SecretStr
    data_root: Path
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8100, ge=1, le=65535)

    steam: SteamConfigSchema
    weblate: WeblateConfigSchema
    limits: WorkshopLimitsSchema = Field(default_factory=_default_limits)
    glossary: GlossaryConfigSchema = Field(default_factory=GlossaryConfigSchema)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="after")
    def validate_loopback(self) -> ServiceConfigSchema:
        if not ipaddress.ip_address(self.bind_host).is_loopback:
            raise ValueError("service must bind to a loopback IP address")
        return self

    @property
    def work_root(self) -> Path:
        return self.data_root / "work"

    @property
    def artifact_root(self) -> Path:
        return self.data_root / "artifacts"
