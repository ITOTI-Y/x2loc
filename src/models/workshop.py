from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from pydantic import Field, field_validator

from src._share import EXT_LANG_MAP
from src.models._share import BaseSchema

XCOM2_APP_ID: Final[int] = 268500
TARGET_LANGUAGE: Final[str] = "zh_Hans"
SOURCE_SUFFIX: Final[str] = f".{EXT_LANG_MAP['en']}"
TARGET_SUFFIX: Final[str] = f".{EXT_LANG_MAP[TARGET_LANGUAGE]}"


class WorkshopLimitsSchema(BaseSchema):
    download_timeout_seconds: float = Field(gt=0)
    terminate_grace_seconds: float = Field(gt=0)
    max_total_bytes: int = Field(gt=0)
    max_file_count: int = Field(gt=0)
    max_loc_file_bytes: int = Field(gt=0)


class WorkshopItemSchema(BaseSchema):
    pass


class LocalizationAssetSchema(BaseSchema):
    source_path: Path
    existing_target_path: Path | None = None
    relative_source_path: PurePosixPath
    relative_target_path: PurePosixPath
    component_slug: str

    @field_validator("relative_source_path", "relative_target_path")
    @classmethod
    def validate_relative_path(cls, value: PurePosixPath) -> PurePosixPath:
        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise ValueError("localization path must stay below the mod root")
        return value

    @property
    def windows_collision_key(self) -> str:
        return PureWindowsPath(self.relative_target_path).as_posix().casefold()
