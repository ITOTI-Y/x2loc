import csv
import io
import json
from pathlib import Path

from src.models.corpus import BilingualCorpus

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


class CorpusWriter:
    def to_csv_string(self, corpus: BilingualCorpus) -> str:
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

    def write_csv(self, corpus: BilingualCorpus, output: Path) -> None:
        """Write corpus to CSV file with UTF-8 BOM (Excel-compatible)."""
        content = self.to_csv_string(corpus)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8-sig")

    def to_json_string(self, corpus: BilingualCorpus) -> str:
        """Serialize corpus to formatted JSON string."""
        return json.dumps(
            corpus.model_dump(mode="json"),
            indent=4,
            ensure_ascii=False,
        )

    def write_json(self, corpus: BilingualCorpus, output: Path) -> None:
        """Write corpus to JSON file."""
        content = self.to_json_string(corpus)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
