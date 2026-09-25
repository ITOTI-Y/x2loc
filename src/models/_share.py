from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

# Agent run defaults, shared by the job request model and both config loaders.
DEFAULT_BATCH_SIZE: Final = 10
DEFAULT_LLM_CONCURRENCY: Final = 10
MAX_LLM_CONCURRENCY: Final = 50


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


class PlaceholderType(StrEnum):
    XGPARAM = "xgparam"
    ABILITY = "ability"
    BULLET = "bullet"
    HEAL = "heal"
    BR = "br"
    HTML = "html"
    PERCENT = "percent"
    PERCENT_WRAPPED = "percent_wrapped"
    NEWLINE = "newline"
    XML_VAR = "xml_var"
    XML_SELF_CLOSE = "xml_self_close"


class SectionHeaderFormat(StrEnum):
    CLASS_ONLY = "class_only"  # [UIUtilities_Text]
    OBJECT_CLASS = "object_class"  # [BattleScanner X2AbilityTemplate]
    PACKAGE_CLASS = "package_class"  # [XComGame.UIFinalShell]
    ARCHETYPE_CLASS = "archetype_class"  # [Archetypes.ARC_xxx ClassName]
