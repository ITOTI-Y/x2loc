from pathlib import Path

from pydantic import Field, computed_field

from src.models._share import BaseSchema


class TermContext(BaseSchema):
    compound_key: str
    section_raw: str
    key: str
    source_path: Path


class GlossaryTerm(BaseSchema):
    source: str
    target: str
    category: str
    do_not_translate: bool = False
    same_as_source: bool = False
    contexts: list[TermContext] = Field(default_factory=list)


class Glossary(BaseSchema):
    source_lang: str
    target_lang: str
    terms: list[GlossaryTerm] = Field(default_factory=list)

    @computed_field
    @property
    def term_count(self) -> int:
        return len(self.terms)
