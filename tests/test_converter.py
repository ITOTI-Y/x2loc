"""Unit tests for CorpusConverter (P3)."""

from pathlib import Path

from src.core.aligner import BilingualAligner
from src.core.converter import (
    TRANSLATABLE_STRUCT_FIELDS,
    CorpusConverter,
)
from src.core.parser import LocFileParser
from src.models._share import SectionHeaderFormat
from src.models.corpus import BilingualCorpus, BilingualEntry
from src.models.entry import EntrySchema, StructFieldSchema
from src.models.section import SectionHeader


def _entry(
    key: str,
    value: str,
    is_append: bool = False,
    struct_fields: list[StructFieldSchema] | None = None,
    raw_value: str | None = None,
) -> EntrySchema:
    return EntrySchema(
        key=key,
        raw_value=raw_value if raw_value is not None else f'"{value}"',
        value=value,
        is_append=is_append,
        struct_fields=struct_fields,
        line_number=1,
    )


def _header(raw: str = "Section") -> SectionHeader:
    return SectionHeader(
        raw=raw,
        format=SectionHeaderFormat.CLASS_ONLY,
        name=raw,
        class_name=raw,
    )


def _bilingual(
    compound_key: str,
    source: EntrySchema,
    target: EntrySchema | None,
    header: SectionHeader | None = None,
) -> BilingualEntry:
    return BilingualEntry(
        compound_key=compound_key,
        source=source,
        target=target,
        section_header=header or _header(),
    )


def _corpus(entries: list[BilingualEntry]) -> BilingualCorpus:
    return BilingualCorpus(
        source_lang="en",
        target_lang="zh_Hans",
        source_path=Path("/src/test.int"),
        target_path=Path("/tgt/test.chn"),
        entries=entries,
    )


class TestToUnitsSimple:
    def test_simple_entry_with_target(self, converter: CorpusConverter) -> None:
        src = _entry("m_strGenericOK", "OK")
        tgt = _entry("m_strGenericOK", "确定")
        corpus = _corpus([_bilingual("Section::m_strGenericOK", src, tgt)])

        units = converter.to_units(corpus)

        assert len(units) == 1
        context, source, target, note = units[0]
        assert context == "Section::m_strGenericOK"
        assert source == "OK"
        assert target == "确定"
        assert note == "section: Section"

    def test_simple_entry_no_target(self, converter: CorpusConverter) -> None:
        src = _entry("Key", "Text")
        corpus = _corpus([_bilingual("Section::Key", src, None)])

        units = converter.to_units(corpus)

        assert len(units) == 1
        assert units[0][2] == ""  # target empty

    def test_skip_empty_source(self, converter: CorpusConverter) -> None:
        src = _entry("Key", "")
        corpus = _corpus([_bilingual("Section::Key", src, None)])

        assert converter.to_units(corpus) == []

    def test_multiple_entries_order_preserved(self, converter: CorpusConverter) -> None:
        e1 = _bilingual("S::A", _entry("A", "foo"), None)
        e2 = _bilingual("S::B", _entry("B", "bar"), None)
        e3 = _bilingual("S::C", _entry("C", "baz"), None)

        units = converter.to_units(_corpus([e1, e2, e3]))

        assert [u[0] for u in units] == ["S::A", "S::B", "S::C"]


class TestToUnitsStructAppend:
    def test_mission_descriptions_expands_description_only(
        self, converter: CorpusConverter
    ) -> None:
        """Real container: +MissionDescriptions=(MissionFamily, Description, MissionIndex).

        Only Description is translatable; MissionFamily is an identifier,
        MissionIndex is a number.
        """
        src_fields = [
            StructFieldSchema(
                key="MissionFamily",
                raw_value='"Recover_LW"',
                value="Recover_LW",
            ),
            StructFieldSchema(
                key="Description",
                raw_value='"We found a cache."',
                value="We found a cache.",
            ),
            StructFieldSchema(
                key="MissionIndex",
                raw_value="0",
                value="0",
            ),
        ]
        tgt_fields = [
            StructFieldSchema(
                key="MissionFamily",
                raw_value='"Recover_LW"',
                value="Recover_LW",
            ),
            StructFieldSchema(
                key="Description",
                raw_value='"我们发现了一个缓存。"',
                value="我们发现了一个缓存。",
            ),
            StructFieldSchema(
                key="MissionIndex",
                raw_value="0",
                value="0",
            ),
        ]
        src = _entry(
            "MissionDescriptions",
            "(MissionFamily=...)",
            is_append=True,
            struct_fields=src_fields,
        )
        tgt = _entry(
            "MissionDescriptions",
            "(MissionFamily=...)",
            is_append=True,
            struct_fields=tgt_fields,
        )
        corpus = _corpus([_bilingual("Section::MissionDescriptions#0", src, tgt)])

        units = converter.to_units(corpus)

        assert len(units) == 1
        context, source, target, note = units[0]
        assert context == "Section::MissionDescriptions#0::Description"
        assert source == "We found a cache."
        assert target == "我们发现了一个缓存。"
        assert "entry: MissionDescriptions" in note

    def test_unit_tactical_info_skips_unit_name(
        self, converter: CorpusConverter
    ) -> None:
        """Real container: +UnitTacticalInfo=(UnitName, UnitDescription).

        UnitName looks like a display name but is actually an X2CharacterTemplate
        DataName ('Muton', 'AdvCaptainM1') and must NOT be translated. Only
        UnitDescription is translatable.
        """
        src_fields = [
            StructFieldSchema(
                key="UnitName",
                raw_value='"Muton"',
                value="Muton",
            ),
            StructFieldSchema(
                key="UnitDescription",
                raw_value='"A member of the alien warrior tribe."',
                value="A member of the alien warrior tribe.",
            ),
        ]
        src = _entry(
            "UnitTacticalInfo",
            "(UnitName=...)",
            is_append=True,
            struct_fields=src_fields,
        )
        corpus = _corpus([_bilingual("DLC::UnitTacticalInfo#0", src, None)])

        units = converter.to_units(corpus)

        assert len(units) == 1
        context, source, _tgt, _note = units[0]
        assert context == "DLC::UnitTacticalInfo#0::UnitDescription"
        assert source == "A member of the alien warrior tribe."
        # UnitName is identifier — must not appear in any unit context
        assert not any("UnitName" in u[0] for u in units)

    def test_animation_pose(self, converter: CorpusConverter) -> None:
        """Real container: +m_arrAnimationPoses=(AnimationDisplayName)."""
        src = _entry(
            "m_arrAnimationPoses",
            "(AnimationDisplayName=...)",
            is_append=True,
            struct_fields=[
                StructFieldSchema(
                    key="AnimationDisplayName",
                    raw_value='"SHIELD Sword 1"',
                    value="SHIELD Sword 1",
                ),
            ],
        )
        corpus = _corpus([_bilingual("DLCInfo::m_arrAnimationPoses#0", src, None)])

        units = converter.to_units(corpus)

        assert len(units) == 1
        assert units[0][0] == ("DLCInfo::m_arrAnimationPoses#0::AnimationDisplayName")

    def test_struct_append_no_translatable_fields(
        self, converter: CorpusConverter
    ) -> None:
        """All fields are identifiers/numbers → no units emitted."""
        src = _entry(
            "Data",
            "(A=1, B=2)",
            is_append=True,
            struct_fields=[
                StructFieldSchema(key="A", raw_value="1", value="1"),
                StructFieldSchema(key="B", raw_value="2", value="2"),
            ],
        )
        corpus = _corpus([_bilingual("S::Data#0", src, None)])

        assert converter.to_units(corpus) == []

    def test_struct_append_empty_field_value_skipped(
        self, converter: CorpusConverter
    ) -> None:
        src = _entry(
            "MissionDescriptions",
            "(Description=...)",
            is_append=True,
            struct_fields=[
                StructFieldSchema(key="Description", raw_value='""', value=""),
            ],
        )
        corpus = _corpus([_bilingual("S::MissionDescriptions#0", src, None)])

        assert converter.to_units(corpus) == []


class TestTranslatableStructFieldsConstant:
    def test_matches_real_data_scan(self) -> None:
        """Regression: whitelist is derived from a full scan of data/.

        See WEBLATE.md §3.5. All three entries come from real mod files:
          Description          — +MissionDescriptions (LW_Overhaul)
          UnitDescription      — +UnitTacticalInfo    (AdditionalUnitInfoWOTC)
          AnimationDisplayName — +m_arrAnimationPoses (WotCBallisticShields)
        """
        assert {
            "Description",
            "UnitDescription",
            "AnimationDisplayName",
        } == TRANSLATABLE_STRUCT_FIELDS

    def test_unit_name_is_not_translatable(self) -> None:
        """Regression: UnitName is a template DataName, not display text.

        +UnitTacticalInfo=(UnitName="Muton", ...) — the string is an
        X2CharacterTemplate identifier; translating it breaks template lookup.
        """
        assert "UnitName" not in TRANSLATABLE_STRUCT_FIELDS


class TestBuildTargetFile:
    def test_simple_translation(
        self,
        converter: CorpusConverter,
        parser: LocFileParser,
        sample_int: Path,
        tmp_path: Path,
    ) -> None:
        source = parser.parse(sample_int)
        translations = {
            "UIUtilities_Text::m_strGenericOK": "确定",
            "UIUtilities_Text::m_strGenericCancel": "取消",
        }

        result = converter.build_target_file(
            source=source,
            translations=translations,
            target_lang="zh_Hans",
            target_path=tmp_path / "sample.chn",
        )

        assert result.lang == "zh_Hans"
        assert result.encoding == "utf-16-le"
        ui_section = next(
            s for s in result.sections if s.header.raw == "UIUtilities_Text"
        )
        ok_entry = next(e for e in ui_section.entries if e.key == "m_strGenericOK")
        assert ok_entry.value == "确定"
        assert ok_entry.raw_value == '"确定"'

        cancel_entry = next(
            e for e in ui_section.entries if e.key == "m_strGenericCancel"
        )
        assert cancel_entry.value == "取消"

    def test_fallback_preserves_source(
        self,
        converter: CorpusConverter,
        parser: LocFileParser,
        sample_int: Path,
        tmp_path: Path,
    ) -> None:
        source = parser.parse(sample_int)
        result = converter.build_target_file(
            source=source,
            translations={},  # no translations provided
            target_lang="zh_Hans",
            target_path=tmp_path / "sample.chn",
        )

        assert result.entry_count == source.entry_count
        for src_sec, new_sec in zip(source.sections, result.sections, strict=True):
            for src_e, new_e in zip(src_sec.entries, new_sec.entries, strict=True):
                assert new_e.value == src_e.value

    def test_unquoted_value_stays_unquoted(
        self, converter: CorpusConverter, tmp_path: Path
    ) -> None:
        """Entries like `KEY=Y` (no quotes) must stay bare after rebuild."""
        from src.models.file import LocalizationFile
        from src.models.section import SectionSchema

        source = LocalizationFile(
            path=tmp_path / "src.int",
            lang="en",
            sections=[
                SectionSchema(
                    header=_header("UIUtilities_Text"),
                    entries=[
                        EntrySchema(
                            key="m_strUpperY",
                            raw_value="Y",
                            value="Y",
                            line_number=1,
                        ),
                    ],
                )
            ],
        )

        result = converter.build_target_file(
            source=source,
            translations={"UIUtilities_Text::m_strUpperY": "Ÿ"},
            target_lang="zh_Hans",
            target_path=tmp_path / "out.chn",
        )
        entry = result.sections[0].entries[0]
        assert entry.value == "Ÿ"
        assert entry.raw_value == "Ÿ"  # no quotes added

    def test_struct_append_partial_translation(
        self,
        converter: CorpusConverter,
        parser: LocFileParser,
        struct_append_int: Path,
        tmp_path: Path,
    ) -> None:
        source = parser.parse(struct_append_int)
        translations = {
            "MissionObjectiveTexts::MissionDescriptions#0::Description": (
                "我们发现了一个补给缓存。"
            ),
        }

        result = converter.build_target_file(
            source=source,
            translations=translations,
            target_lang="zh_Hans",
            target_path=tmp_path / "out.chn",
        )

        mission_section = result.sections[0]
        entry0 = mission_section.entries[0]
        assert entry0.is_append
        assert entry0.struct_fields is not None

        desc_field = next(f for f in entry0.struct_fields if f.key == "Description")
        assert desc_field.value == "我们发现了一个补给缓存。"
        assert desc_field.raw_value == '"我们发现了一个补给缓存。"'

        # MissionIndex (non-translatable) unchanged, bare integer
        idx_field = next(f for f in entry0.struct_fields if f.key == "MissionIndex")
        assert idx_field.raw_value == "0"

        # raw_value has been re-rendered with new Description
        assert "我们发现了一个补给缓存。" in entry0.raw_value
        assert "MissionIndex=0" in entry0.raw_value
        assert 'MissionFamily="Recover_LW"' in entry0.raw_value

        # Entry #1 (no translation) is preserved verbatim
        entry1 = mission_section.entries[1]
        assert entry1.struct_fields is not None
        desc1 = next(f for f in entry1.struct_fields if f.key == "Description")
        assert desc1.value == "Rescue the VIP."

    def test_struct_append_untouched_when_nothing_translated(
        self,
        converter: CorpusConverter,
        parser: LocFileParser,
        struct_append_int: Path,
        tmp_path: Path,
    ) -> None:
        """When no field of a struct append is translated, entry stays intact."""
        source = parser.parse(struct_append_int)

        result = converter.build_target_file(
            source=source,
            translations={},
            target_lang="zh_Hans",
            target_path=tmp_path / "out.chn",
        )
        # First entry's struct fields unchanged
        entry = result.sections[0].entries[0]
        assert entry.struct_fields is not None
        desc = next(f for f in entry.struct_fields if f.key == "Description")
        assert desc.value == "We have discovered a supply cache."

    def test_comments_dropped_from_target(
        self,
        converter: CorpusConverter,
        parser: LocFileParser,
        sample_int: Path,
        tmp_path: Path,
    ) -> None:
        source = parser.parse(sample_int)
        # sample_int has header_comments and section-level comments
        assert source.header_comments  # sanity

        result = converter.build_target_file(
            source=source,
            translations={},
            target_lang="zh_Hans",
            target_path=tmp_path / "out.chn",
        )
        assert result.header_comments == []
        for sec in result.sections:
            assert sec.comments == []


class TestCompoundKeyParity:
    """Converter and aligner must agree on compound_key format."""

    def test_context_keys_line_up_with_aligner_output(
        self,
        converter: CorpusConverter,
        parser: LocFileParser,
        aligner: BilingualAligner,
        struct_append_int: Path,
        struct_append_chn: Path,
    ) -> None:
        src_file = parser.parse(struct_append_int)
        tgt_file = parser.parse(struct_append_chn)
        corpus = aligner.align(src_file, tgt_file)

        entry_keys = {e.compound_key for e in corpus.entries}

        for context, _src, _tgt, _note in converter.to_units(corpus):
            parts = context.split("::")
            # struct field contexts end in ::{FieldKey}; strip to get parent
            if len(parts) >= 3 and parts[-1] in TRANSLATABLE_STRUCT_FIELDS:
                parent = "::".join(parts[:-1])
            else:
                parent = context
            assert parent in entry_keys, f"orphan context: {context}"
