from typing import Annotated, Final, Literal, Self

from pydantic import BeforeValidator, ConfigDict, Field, SecretStr

from src.models._share import BaseSchema

UNIT_PAGE_SIZE: Final[int] = 100


def _list_to_str(value: list[str] | str) -> str:
    return "".join(value) if isinstance(value, list) else value


ListAsStr = Annotated[str, BeforeValidator(_list_to_str)]


class WeblateConfigSchema(BaseSchema):
    url: str
    token: SecretStr
    project_slug: str
    license: str = "CC-BY-4.0"
    license_url: str = "https://creativecommons.org/licenses/by/4.0/"
    source_language: str = "en"


class WeblateRequestParamsSchema(BaseSchema):
    page: int = 1
    page_size: int = UNIT_PAGE_SIZE
    q: str = ""

    def to_query(self) -> dict[str, str | int]:
        query: dict[str, str | int] = {"page": self.page, "page_size": self.page_size}
        return query | ({"q": self.q} if self.q else {})


class WeblateRequestSchema(BaseSchema):
    """One Weblate API call: everything `_request` needs to build it."""

    method: Literal["GET", "POST", "PATCH", "DELETE"]
    path: str
    params: WeblateRequestParamsSchema | None = None
    json_body: dict[str, object] | None = None
    data: dict[str, str] | None = None
    files: dict[str, tuple[str, bytes, str]] | None = None

    def query_params(self) -> dict[str, str | int] | None:
        return self.params.to_query() if self.params else None


class WeblateComponentDraftSchema(BaseSchema):
    """Payload for VCS-less component creation from a source docfile."""

    name: str
    slug: str
    source_csv: bytes


class WeblateUnitDraftSchema(BaseSchema):
    """Payload for creating one source string on a template component.

    Weblate only accepts unit creation on the source translation, in the
    monolingual key/value shape; targets are filled afterwards through
    translate uploads.
    """

    context: str
    source: str


class WeblateUnitPatchSchema(BaseSchema):
    model_config = ConfigDict(extra="forbid")

    target: list[str]
    state: int


class CorpusUnitSchema(BaseSchema):
    """One aligned corpus row bound for Weblate CSV upload."""

    context: str
    source: str
    target: str
    note: str

    @classmethod
    def from_row(cls, row: tuple[str, str, str, str]) -> Self:
        context, source, target, note = row
        return cls(context=context, source=source, target=target, note=note)


class WeblateComponentSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore", frozen=True)

    slug: str
    name: str = ""
    file_format: str = ""
    manage_units: bool = False
    edit_template: bool = False


class WeblateUnitSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int
    translation: str = ""
    language_code: str
    source: ListAsStr
    target: ListAsStr
    context: str
    source_unit: str = ""
    web_url: str = ""
    url: str = ""
    position: int = 0
    note: str | None = None
    state: int = 0


class WeblatePageSchema[T: BaseSchema](BaseSchema):
    model_config = ConfigDict(extra="ignore", frozen=True)

    results: list[T] = Field(default_factory=list)
    next: str | None = None
    count: int = 0
    page: int | None = None


class WeblateTaskResultSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore")

    error: str | None = None


class WeblateTaskSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore")

    completed: bool = False
    progress: int = 0
    result: WeblateTaskResultSchema | None = None


class WeblateUploadResultSchema(BaseSchema):
    model_config = ConfigDict(extra="ignore")

    task_url: str | None = None
    accepted: int = 0
    skipped: int = 0
    not_found: int = 0
