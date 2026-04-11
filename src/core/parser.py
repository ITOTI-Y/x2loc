from pathlib import Path

from loguru import logger

from src._share import LANG_EXT_MAP
from src.core._share import (
    ARRAY_INDEX_RE,
    COMMENT_PREFIX,
    ENTRY_RE,
    PLACEHOLDER_PATTERNS,
    SECTION_RE,
)
from src.models._share import SectionHeaderFormat
from src.models.entry import EntrySchema, PlaceholderSchema, StructFieldSchema
from src.models.file import LocalizationFile
from src.models.section import SectionHeader, SectionSchema


class LocFileParser:
    def parse(self, path: Path, lang_override: str | None = None) -> LocalizationFile:
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        encoding = self._infer_encoding(path)

        lang = lang_override or self._infer_lang(path)

        lines = self._read_file(path, encoding)

        sections, header_comments = self._scan_lines(lines, path)

        return LocalizationFile(
            path=path,
            lang=lang,
            encoding=encoding,
            header_comments=header_comments,
            sections=sections,
        )

    def _infer_encoding(self, path: Path) -> str:
        raw = path.read_bytes()
        head = raw[:4]
        match head:
            case _ if head.startswith(b"\xef\xbb\xbf"):
                return "utf-8-sig"
            case _ if head.startswith(b"\xff\xfe"):
                return "utf-16-le"
            case _ if head.startswith(b"\xfe\xff"):
                return "utf-16-be"
            case _:
                try:
                    raw.decode("utf-8")
                    return "utf-8"
                except UnicodeDecodeError:
                    logger.warning(
                        f"{path.name}: not valid UTF-8, falling back to latin1"
                    )
                    return "latin1"

    def _read_file(self, path: Path, encoding: str) -> list[str]:
        with open(path, encoding=encoding, errors="strict") as f:
            content = f.read()

        if content.startswith("\ufeff"):
            content = content[1:]

        return content.splitlines()

    def _infer_lang(self, path: Path) -> str:
        ext = path.suffix.lstrip(".").lower()
        lang = LANG_EXT_MAP.get(ext)
        if not lang:
            raise ValueError(f"Unsupported file extension: {ext}")
        return lang

    def _scan_lines(
        self, lines: list[str], path: Path
    ) -> tuple[list[SectionSchema], list[str]]:
        current_section: SectionSchema | None = None
        pending_comments: list[str] = []
        sections: list[SectionSchema] = []
        header_comments: list[str] = []
        seen_first_section = False

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            if not line or line == "\ufeff":
                continue

            if line.startswith(COMMENT_PREFIX):
                pending_comments.append(line[1:].strip())
                continue

            m_section = SECTION_RE.match(line)
            if m_section:
                if current_section is not None:
                    sections.append(current_section)

                if not seen_first_section:
                    header_comments = pending_comments
                    seen_first_section = True
                    section_comments = []
                else:
                    section_comments = pending_comments

                header = self._parse_section_header(m_section.group(1))
                current_section = SectionSchema(
                    header=header,
                    comments=section_comments,
                    entries=[],
                )
                pending_comments = []
                continue

            m_entry = ENTRY_RE.match(line)
            if m_entry:
                if current_section is None:
                    # UE3 engine silently discards entries before any section
                    logger.warning(
                        f"{path.name}:{line_number}: "
                        "Entry before any section header (discarded)"
                    )
                    continue
                entry = self._parse_entry(
                    m_entry.groups(),  # type: ignore
                    line_number,
                    pending_comments,
                )

                pending_comments = []
                current_section.entries.append(entry)
                continue

            logger.warning(f"Unrecognized line {line_number}: {line[:80]}")

        if current_section is not None:
            sections.append(current_section)
        return sections, header_comments

    def _parse_section_header(self, content: str) -> SectionHeader:
        """Parse the content inside [] into a SectionHeader.

        Classifies into one of four formats:
            A: class_only       — [UIUtilities_Text]
            B: object_class     — [BattleScanner X2AbilityTemplate]
            C: package_class    — [XComGame.UIFinalShell]
            D: archetype_class  — [Archetypes.ARC_xxx ClassName]
        """
        content = content.strip()
        tokens = content.split(None, 1)

        match tokens:
            case [single] if "." in single:
                # Format C: "XComGame.UIFinalShell"
                package, class_name = single.rsplit(".", 1)
                return SectionHeader(
                    raw=content,
                    format=SectionHeaderFormat.PACKAGE_CLASS,
                    name=content,
                    class_name=class_name,
                    package=package,
                )

            case [single]:
                # Format A: "UIUtilities_Text"
                return SectionHeader(
                    raw=content,
                    format=SectionHeaderFormat.CLASS_ONLY,
                    name=single,
                    class_name=single,
                )

            case [first, rest] if "." in first:
                # Format D: "Archetypes.ARC_xxx ClassName"
                package, object_part = first.rsplit(".", 1)
                return SectionHeader(
                    raw=content,
                    format=SectionHeaderFormat.ARCHETYPE_CLASS,
                    name=object_part,
                    object_name=object_part,
                    class_name=rest.strip(),
                    package=package,
                )

            case [first, rest]:
                # Format B: "BattleScanner X2AbilityTemplate"
                return SectionHeader(
                    raw=content,
                    format=SectionHeaderFormat.OBJECT_CLASS,
                    name=first,
                    object_name=first,
                    class_name=rest.strip(),
                )

            case _:
                # Defensive fallback — should not happen
                logger.warning(
                    f"Could not classify section header '{content}', "
                    "defaulting to CLASS_ONLY"
                )
                return SectionHeader(
                    raw=content,
                    format=SectionHeaderFormat.CLASS_ONLY,
                    name=content,
                )

    def _parse_entry(
        self, groups: tuple[str, str, str], line_number: int, comments: list[str]
    ) -> EntrySchema:
        plus_prefix, key_with_index, raw_value = groups

        is_append = plus_prefix == "+"

        m_array = ARRAY_INDEX_RE.search(key_with_index)
        if m_array:
            is_array = True
            idx_str = m_array.group(1)
            array_index = int(idx_str) if idx_str.isdigit() else None
        else:
            is_array = False
            array_index = None

        key = key_with_index

        raw_value_stripped = raw_value.strip()
        if (
            len(raw_value_stripped) >= 2
            and raw_value_stripped.startswith('"')
            and raw_value_stripped.endswith('"')
        ):
            value = raw_value_stripped[1:-1]
        elif raw_value_stripped.startswith('"'):
            # Unclosed string literal: author forgot the closing `"`.
            # Observed in real mods (e.g. RealModFiles/2867288932 T2/T3/T4
            # weapon templates with `AbilityDescName="..."` missing the
            # trailing quote). Strip the leading `"` alone so downstream
            # Weblate upload and glossary extraction see the intended text
            # rather than leaking the stray quote into translator views.
            logger.warning(
                f"Unclosed string literal at line {line_number}: "
                f"{raw_value_stripped[:80]}"
            )
            value = raw_value_stripped[1:]
        elif raw_value_stripped.endswith('"'):
            # Mirror case: trailing `"` without an opening one. Less
            # common but still observed (mods with typos like
            # `Key=value"`). Strip the trailing orphan.
            logger.warning(
                f"Unopened string literal at line {line_number}: "
                f"{raw_value_stripped[:80]}"
            )
            value = raw_value_stripped[:-1]
        else:
            value = raw_value_stripped

        placeholders = self._extract_placeholders(value)

        struct_fields: list[StructFieldSchema] | None = None
        if is_append and value.startswith("(") and value.endswith(")"):
            struct_fields = self._parse_struct_fields(value[1:-1])

        return EntrySchema(
            key=key,
            raw_value=raw_value,
            value=value,
            is_array=is_array,
            array_index=array_index,
            is_append=is_append,
            struct_fields=struct_fields,
            placeholders=placeholders,
            comments=comments,
            line_number=line_number,
        )

    def _extract_placeholders(self, value: str) -> list[PlaceholderSchema]:
        """Extract all placeholder tokens from a value string.

        Iterates PLACEHOLDER_PATTERNS in priority order.
        Deduplicates by span (first matching pattern wins).
        Returns results sorted by span start position.
        """

        occupied_spans: set[tuple[int, int]] = set()
        results: list[PlaceholderSchema] = []
        for pattern, ph_type in PLACEHOLDER_PATTERNS:
            for m in pattern.finditer(value):
                span = (m.start(), m.end())
                if any(
                    span[0] < occ_end and span[1] > occ_start
                    for occ_start, occ_end in occupied_spans
                ):
                    continue

                occupied_spans.add(span)
                results.append(
                    PlaceholderSchema(
                        pattern=m.group(),
                        type=ph_type,
                        span=span,
                    )
                )
        results.sort(key=lambda p: p.span[0])
        return results

    def _parse_struct_fields(self, content: str) -> list[StructFieldSchema]:
        """Parse the inside of (...) struct literal into individual fields.

        Handles quoted strings containing commas correctly via state scanning.
        """
        fields_raw: list[str] = []
        current: list[str] = []
        in_quotes = False

        for c in content:
            if in_quotes:
                current.append(c)
                if c == '"':
                    in_quotes = False
            elif c == '"':
                in_quotes = True
                current.append(c)
            elif c == ",":
                fields_raw.append("".join(current).strip())
                current = []
            else:
                current.append(c)

        tail = "".join(current).strip()
        if tail:
            fields_raw.append(tail)

        # Parse each "Key=Value" field
        result: list[StructFieldSchema] = []
        for field_str in fields_raw:
            if "=" not in field_str:
                logger.warning(f"Struct field without '=': '{field_str[:60]}'")
                continue

            field_key, field_raw_value = field_str.split("=", 1)
            field_key = field_key.strip()
            field_raw_value = field_raw_value.strip()

            # Strip quotes for the cleaned value
            if (
                len(field_raw_value) >= 2
                and field_raw_value.startswith('"')
                and field_raw_value.endswith('"')
            ):
                field_value = field_raw_value[1:-1]
                field_placeholders = self._extract_placeholders(field_value)
            else:
                field_value = field_raw_value
                field_placeholders = []  # numeric/non-string — no placeholders

            result.append(
                StructFieldSchema(
                    key=field_key,
                    raw_value=field_raw_value,
                    value=field_value,
                    placeholders=field_placeholders,
                )
            )

        return result
