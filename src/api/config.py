from __future__ import annotations

import ipaddress
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models._share import BaseSchema
from src.models.weblate import WeblateConfigSchema
from src.models.workshop import WorkshopLimitsSchema


class SteamConfigSchema(BaseSchema):
    executable: Path
    root: Path
    username: str
    password: SecretStr


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
    """Loaded from `X2LOC_`-prefixed environment variables only.

    Credentials come from the process environment; the repository holds no
    `.env` or credential template files. Nested fields use `__`, e.g.
    `X2LOC_WEBLATE__TOKEN` and `X2LOC_STEAM__PASSWORD`.
    """

    model_config = SettingsConfigDict(
        env_prefix="X2LOC_",
        env_nested_delimiter="__",
        env_file=None,
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
