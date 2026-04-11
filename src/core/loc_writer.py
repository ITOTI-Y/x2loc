from pathlib import Path
from typing import Final

from src.models.entry import EntrySchema
from src.models.file import LocalizationFile

LINE_TERMINATOR: Final[str] = "\r\n"
BOM: Final[bytes] = b"\xff\xfe"
ENCODING: Final[str] = "utf-16-le"


class LocFileWriter:
    """Serialize a LocalizationFile back to on-disk UE3 loc format.

    Format contract:
        - UTF-16-LE with BOM (\\xff\\xfe)
        - CRLF line endings
        - "[{section.raw}]" section header lines
        - "{key}={raw_value}" / "+{key}={raw_value}" entries
        - one blank line between sections, none before the first section
        - source-side comments dropped (output is a translation artifact)
    """

    def write(self, file: LocalizationFile, path: Path) -> None:
        text = self.to_text(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(BOM + text.encode(ENCODING))

    def to_text(self, file: LocalizationFile) -> str:
        lines: list[str] = []

        for idx, section in enumerate(file.sections):
            if idx > 0:
                lines.append("")  # blank separator between sections

            lines.append(f"[{section.header.raw}]")
            for entry in section.entries:
                lines.append(self._format_entry(entry))

        return LINE_TERMINATOR.join(lines) + LINE_TERMINATOR

    def _format_entry(self, entry: EntrySchema) -> str:
        prefix = "+" if entry.is_append else ""
        return f"{prefix}{entry.key}={entry.raw_value}"
