import csv
import io
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.models.corpus import BilingualCorpus
from src.models.file import LocalizationFile
from src.models.glossary import Glossary

type OutputFormat = Literal["csv", "json"]


CSV_COLUMNS: list[str] = [
    "compound_key",
    "section",
    "key",
    "is_append",
    "source_value",
    "target_value",
    "source_has_placeholders",
    "target_has_placeholders",
    "source_line",
    "target_line",
    "status",
]

GLOSSARY_CSV_COLUMNS: list[str] = [
    "source",
    "target",
    "category",
    "context_section",
    "context_key",
    "context_source_file",
    "do_not_translate",
    "same_as_source",
    "context_count",
]


def corpus_to_csv(corpus: BilingualCorpus) -> str:
    source_only_set = set(corpus.source_only)
    target_only_set = set(corpus.target_only)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for entry in corpus.entries:
        if entry.compound_key in target_only_set:
            status = "target_only"
        elif entry.compound_key in source_only_set:
            status = "source_only"
        else:
            status = "aligned"
        row = {
            "compound_key": entry.compound_key,
            "section": entry.section_header.raw,
            "key": entry.source.key,
            "is_append": entry.source.is_append,
            "source_value": entry.source.value,
            "target_value": entry.target.value if entry.target else "",
            "source_has_placeholders": bool(entry.source.placeholders),
            "target_has_placeholders": bool(entry.target.placeholders)
            if entry.target
            else "",
            "source_line": entry.source.line_number,
            "target_line": entry.target.line_number if entry.target else "",
            "status": status,
        }
        writer.writerow(row)

    return output.getvalue()


def glossary_to_csv(glossary: Glossary) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=GLOSSARY_CSV_COLUMNS)
    writer.writeheader()

    for term in glossary.terms:
        first_ctx = term.contexts[0] if term.contexts else None
        row = {
            "source": term.source,
            "target": term.target,
            "category": term.category,
            "context_section": first_ctx.section_raw if first_ctx else "",
            "context_key": first_ctx.key if first_ctx else "",
            "context_source_file": str(first_ctx.source_path) if first_ctx else "",
            "do_not_translate": "true" if term.do_not_translate else "",
            "same_as_source": "true" if term.same_as_source else "",
            "context_count": len(term.contexts),
        }
        writer.writerow(row)

    return output.getvalue()


LOC_FILE_CSV_COLUMNS: list[str] = [
    "section",
    "key",
    "value",
    "is_array",
    "is_append",
    "line_number",
    "has_placeholders",
]


def loc_file_to_csv(loc_file: LocalizationFile) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=LOC_FILE_CSV_COLUMNS)
    writer.writeheader()
    for section in loc_file.sections:
        for entry in section.entries:
            writer.writerow(
                {
                    "section": section.header.raw,
                    "key": entry.key,
                    "value": entry.value,
                    "is_array": entry.is_array,
                    "is_append": entry.is_append,
                    "line_number": entry.line_number,
                    "has_placeholders": bool(entry.placeholders),
                }
            )
    return output.getvalue()


def to_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=4, ensure_ascii=False)


def write_text(content: str, output: Path, output_format: OutputFormat) -> None:
    """Write serialized output; CSV carries a UTF-8 BOM for Excel/Weblate."""
    encoding = "utf-8-sig" if output_format == "csv" else "utf-8"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding=encoding)
