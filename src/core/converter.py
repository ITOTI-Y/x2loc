from pathlib import Path
from typing import Final

from src.core._share import iter_compound_keys_in_section
from src.models.corpus import BilingualCorpus
from src.models.entry import EntrySchema, StructFieldSchema
from src.models.file import LocalizationFile
from src.models.section import SectionSchema

TRANSLATABLE_STRUCT_FIELDS: Final[set[str]] = {
    "Description",  # +MissionDescriptions=(...) Mission Description
    "UnitDescription",  # +UnitTacticalInfo=(...) Unit Description
    "AnimationDisplayName",  # +m_arrAnimationPoses=(...) Animation Display Name
}


class CorpusConverter:
    """Bidirectional bridge between BilingualCorpus and Weblate translation units.

    to_units:          BilingualCorpus  → [(context, source, target, note), ...]
    build_target_file: LocalizationFile + translations dict → new LocalizationFile
    """

    def to_units(self, corpus: BilingualCorpus) -> list[tuple[str, str, str, str]]:
        """Convert a bilingual corpus to Weblate translation-unit tuples.

        Target-only entries (keys present only in the target file, not the
        source) are **skipped**: their `source` field has been synthesized
        from the target entry by the aligner, which means `source.value`
        would be the target-language text — uploading that to Weblate as the
        English source string pollutes the source corpus. These entries are
        translation orphans and have no valid round-trip path.

        Returns:
            List of (context, source, target, developer_comments) tuples,
            ready for CSV serialization.
        """
        target_only_keys = set(corpus.target_only)
        units: list[tuple[str, str, str, str]] = []

        for entry in corpus.entries:
            if entry.compound_key in target_only_keys:
                continue  # orphan — no valid source language text

            src = entry.source
            tgt = entry.target

            if not src.value:
                continue

            if not src.is_append or src.struct_fields is None:
                target_value = tgt.value if tgt is not None else ""
                note = f"section: {entry.section_header.raw}"
                units.append((entry.compound_key, src.value, target_value, note))
                continue

            tgt_field_by_key: dict[str, str] = {}
            if tgt is not None and tgt.struct_fields is not None:
                tgt_field_by_key = {f.key: f.value for f in tgt.struct_fields}

            for field in src.struct_fields:
                if field.key not in TRANSLATABLE_STRUCT_FIELDS:
                    continue
                if not field.value:
                    continue

                context = f"{entry.compound_key}::{field.key}"
                target_field_value = tgt_field_by_key.get(field.key, "")
                note = f"section: {entry.section_header.raw}, entry: {src.key}"
                units.append((context, field.value, target_field_value, note))
        return units

    def build_target_file(
        self,
        source: LocalizationFile,
        translations: dict[str, str],
        target_lang: str,
        target_path: Path,
    ) -> LocalizationFile:
        """Build a target LocalizationFile by applying translations to source.

        Args:
            source: Parsed source .int file (structural template).
            translations: context → translated value mapping.
            target_lang: BCP-47 target language code.
            target_path: Output file path (metadata only, no I/O performed).

        Returns:
            New LocalizationFile carrying translated values. Missing
            translations fall back to source values.
        """
        new_sections: list[SectionSchema] = []

        for section in source.sections:
            new_entries: list[EntrySchema] = [
                self._rebuild_entry(entry, compound_key, translations)
                for compound_key, entry in iter_compound_keys_in_section(section)
            ]
            new_sections.append(
                SectionSchema(
                    header=section.header,
                    entries=new_entries,
                    comments=[],  # drop source-side comments in the output
                )
            )

        return LocalizationFile(
            path=target_path,
            lang=target_lang,
            encoding="utf-16-le",
            header_comments=[],
            sections=new_sections,
        )

    def _rebuild_entry(
        self,
        entry: EntrySchema,
        compound_key: str,
        translations: dict[str, str],
    ) -> EntrySchema:
        """Dispatch to struct-field vs simple rebuild path."""
        if entry.is_append and entry.struct_fields is not None:
            return self._rebuild_struct_entry(entry, compound_key, translations)
        return self._rebuild_simple_entry(entry, compound_key, translations)

    def _rebuild_simple_entry(
        self,
        entry: EntrySchema,
        compound_key: str,
        translations: dict[str, str],
    ) -> EntrySchema:
        translated = translations.get(compound_key)
        if translated is None:
            return entry.model_copy()

        new_value = translated
        new_raw_value = _restore_quoting(entry.raw_value, new_value)
        return entry.model_copy(
            update={
                "value": new_value,
                "raw_value": new_raw_value,
                "placeholders": [],  # placeholders recomputation is not needed
                # for writeback, writer only uses raw_value
                "comments": [],
            }
        )

    def _rebuild_struct_entry(
        self,
        entry: EntrySchema,
        compound_key: str,
        translations: dict[str, str],
    ) -> EntrySchema:
        assert entry.struct_fields is not None
        new_fields: list[StructFieldSchema] = []
        any_translated = False

        for field in entry.struct_fields:
            if field.key in TRANSLATABLE_STRUCT_FIELDS:
                field_context = f"{compound_key}::{field.key}"
                translated = translations.get(field_context)
                if translated is not None:
                    any_translated = True
                    new_fields.append(
                        field.model_copy(
                            update={
                                "value": translated,
                                "raw_value": _restore_quoting(
                                    field.raw_value, translated
                                ),
                                "placeholders": [],
                            }
                        )
                    )
                    continue
            new_fields.append(field.model_copy())

        if not any_translated:
            return entry.model_copy(update={"comments": []})

        new_value = _render_struct_value(new_fields)
        return entry.model_copy(
            update={
                "struct_fields": new_fields,
                "value": new_value,
                "raw_value": new_value,
                "placeholders": [],
                "comments": [],
            }
        )


def _restore_quoting(original_raw: str, new_value: str) -> str:
    """Recreate the raw_value using the same quoting style as the original.

    Rule: if original_raw starts with `"`, wrap new_value in double quotes;
    otherwise return new_value bare.
    """
    stripped = original_raw.strip()
    if stripped.startswith('"'):
        return f'"{new_value}"'
    return new_value


def _render_struct_value(fields: list[StructFieldSchema]) -> str:
    """Re-serialize a struct field list into the `(k1=v1, k2=v2, ...)` form."""
    parts = [f"{f.key}={f.raw_value}" for f in fields]
    return "(" + ", ".join(parts) + ")"
