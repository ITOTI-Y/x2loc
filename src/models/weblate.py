from typing import Annotated

from pydantic import BeforeValidator, ConfigDict, Field

from src.models._share import BaseSchema


def _list_to_str(v: list[str] | str) -> str:
    if isinstance(v, list):
        return "".join(v)
    return v

ListAsStr = Annotated[str, BeforeValidator(_list_to_str)]

class WeblateConfigSchema(BaseSchema):
    url: str
    token: str
    project_slug: str
    license: str = "CC-BY-4.0"
    license_url: str = "https://creativecommons.org/licenses/by/4.0/"

class WeblateRequestParamsSchema(BaseSchema):
    page: int | None = None
    page_size: int = 100
    q: str | None = None

class WeblateRequestSchema(BaseSchema):
    path: str
    params: WeblateRequestParamsSchema

class WeblateUnitSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int
    translation: str
    language_code: str
    source: ListAsStr
    target: ListAsStr
    context: str
    source_unit: str
    web_url: str
    url: str
    position: int
    note: str | None = None

class WeblatePageSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore", frozen=True)

    units: list[WeblateUnitSchema] = Field(default_factory=list, alias="results")
    next: str | None
    count: int | None = None
    page: int | None = None

class WeblateTranslationSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int
    total: int
    translated_percent: float
    total_words: int
    translated_words: int
    url: str
    translate_url: str
