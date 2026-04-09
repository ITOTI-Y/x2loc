from enum import StrEnum

from pydantic import BaseModel, ConfigDict


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
