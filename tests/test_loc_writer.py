"""Unit tests for LocFileWriter and a parse→build→write→re-parse round-trip."""

from pathlib import Path

from src.core.converter import CorpusConverter
from src.core.loc_writer import BOM, ENCODING, LocFileWriter
from src.core.parser import LocFileParser
from src.models._share import SectionHeaderFormat
from src.models.entry import EntrySchema
from src.models.file import LocalizationFile
from src.models.section import SectionHeader, SectionSchema


def _two_section_file(tmp_path: Path) -> LocalizationFile:
    return LocalizationFile(
        path=tmp_path / "out.chn",
        lang="zh_Hans",
        sections=[
            SectionSchema(
                header=SectionHeader(
                    raw="UIUtilities_Text",
                    format=SectionHeaderFormat.CLASS_ONLY,
                    name="UIUtilities_Text",
                    class_name="UIUtilities_Text",
                ),
                entries=[
                    EntrySchema(
                        key="m_strGenericOK",
                        raw_value='"确定"',
                        value="确定",
                        line_number=1,
                    ),
                    EntrySchema(
                        key="m_strGenericCancel",
                        raw_value='"取消"',
                        value="取消",
                        line_number=2,
                    ),
                ],
            ),
            SectionSchema(
                header=SectionHeader(
                    raw="BattleScanner X2AbilityTemplate",
                    format=SectionHeaderFormat.OBJECT_CLASS,
                    name="BattleScanner",
                    object_name="BattleScanner",
                    class_name="X2AbilityTemplate",
                ),
                entries=[
                    EntrySchema(
                        key="LocFriendlyName",
                        raw_value='"战场扫描器"',
                        value="战场扫描器",
                        line_number=3,
                    ),
                ],
            ),
        ],
    )


class TestLocFileWriter:
    def test_utf16le_bom(self, loc_writer: LocFileWriter, tmp_path: Path) -> None:
        file = _two_section_file(tmp_path)
        out = tmp_path / "out.chn"
        loc_writer.write(file, out)

        raw = out.read_bytes()
        assert raw.startswith(BOM)

    def test_crlf_line_endings(self, loc_writer: LocFileWriter, tmp_path: Path) -> None:
        file = _two_section_file(tmp_path)
        out = tmp_path / "out.chn"
        loc_writer.write(file, out)

        text = out.read_bytes()[len(BOM) :].decode(ENCODING)
        # CRLF present
        assert "\r\n" in text
        # No lone \n without preceding \r
        lines = text.split("\r\n")
        for line in lines:
            assert "\n" not in line

    def test_section_header_format(
        self, loc_writer: LocFileWriter, tmp_path: Path
    ) -> None:
        file = _two_section_file(tmp_path)
        text = loc_writer.to_text(file)
        assert "[UIUtilities_Text]" in text
        assert "[BattleScanner X2AbilityTemplate]" in text

    def test_section_separator(self, loc_writer: LocFileWriter, tmp_path: Path) -> None:
        file = _two_section_file(tmp_path)
        text = loc_writer.to_text(file)
        lines = text.split("\r\n")
        # First line must be a section header (no leading blank)
        assert lines[0] == "[UIUtilities_Text]"
        # Between last entry of section 1 and header of section 2 there is
        # exactly one blank line
        idx_first_header = lines.index("[UIUtilities_Text]")
        idx_second_header = lines.index("[BattleScanner X2AbilityTemplate]")
        between = lines[idx_first_header + 1 : idx_second_header]
        assert between.count("") == 1

    def test_simple_entry_format(
        self, loc_writer: LocFileWriter, tmp_path: Path
    ) -> None:
        file = _two_section_file(tmp_path)
        text = loc_writer.to_text(file)
        assert 'm_strGenericOK="确定"' in text
        assert 'm_strGenericCancel="取消"' in text

    def test_append_prefix(self, loc_writer: LocFileWriter, tmp_path: Path) -> None:
        file = LocalizationFile(
            path=tmp_path / "out.chn",
            lang="zh_Hans",
            sections=[
                SectionSchema(
                    header=SectionHeader(
                        raw="Section",
                        format=SectionHeaderFormat.CLASS_ONLY,
                        name="Section",
                        class_name="Section",
                    ),
                    entries=[
                        EntrySchema(
                            key="Mission",
                            raw_value='(A="a")',
                            value='(A="a")',
                            is_append=True,
                            line_number=1,
                        ),
                    ],
                ),
            ],
        )
        text = loc_writer.to_text(file)
        assert '+Mission=(A="a")' in text

    def test_array_key_preserved(
        self, loc_writer: LocFileWriter, tmp_path: Path
    ) -> None:
        file = LocalizationFile(
            path=tmp_path / "out.chn",
            lang="zh_Hans",
            sections=[
                SectionSchema(
                    header=SectionHeader(
                        raw="S",
                        format=SectionHeaderFormat.CLASS_ONLY,
                        name="S",
                        class_name="S",
                    ),
                    entries=[
                        EntrySchema(
                            key="RankNames[0]",
                            raw_value='"新兵"',
                            value="新兵",
                            is_array=True,
                            array_index=0,
                            line_number=1,
                        ),
                    ],
                ),
            ],
        )
        text = loc_writer.to_text(file)
        assert 'RankNames[0]="新兵"' in text

    def test_empty_file(self, loc_writer: LocFileWriter, tmp_path: Path) -> None:
        file = LocalizationFile(
            path=tmp_path / "out.chn",
            lang="zh_Hans",
            sections=[],
        )
        out = tmp_path / "out.chn"
        loc_writer.write(file, out)
        raw = out.read_bytes()
        assert raw.startswith(BOM)
        # Just BOM + terminator, no content
        assert len(raw) < 20

    def test_write_creates_parent_dir(
        self, loc_writer: LocFileWriter, tmp_path: Path
    ) -> None:
        out = tmp_path / "deep" / "nested" / "dir" / "out.chn"
        file = _two_section_file(tmp_path)
        loc_writer.write(file, out)
        assert out.exists()
        assert out.read_bytes().startswith(BOM)

    def test_comments_not_emitted(
        self, loc_writer: LocFileWriter, tmp_path: Path
    ) -> None:
        """Output is a translation artifact — source-side comments dropped."""
        file = LocalizationFile(
            path=tmp_path / "out.chn",
            lang="zh_Hans",
            header_comments=["Do not emit this"],
            sections=[
                SectionSchema(
                    header=SectionHeader(
                        raw="S",
                        format=SectionHeaderFormat.CLASS_ONLY,
                        name="S",
                        class_name="S",
                    ),
                    entries=[
                        EntrySchema(
                            key="K",
                            raw_value='"V"',
                            value="V",
                            comments=["nor this"],
                            line_number=1,
                        ),
                    ],
                    comments=["nor this either"],
                ),
            ],
        )
        text = loc_writer.to_text(file)
        assert "Do not emit this" not in text
        assert "nor this" not in text


class TestRoundTrip:
    def test_parse_build_write_reparse_struct_append(
        self,
        parser: LocFileParser,
        converter: CorpusConverter,
        loc_writer: LocFileWriter,
        struct_append_int: Path,
        tmp_path: Path,
    ) -> None:
        source = parser.parse(struct_append_int)
        translations = {
            "MissionObjectiveTexts::MissionDescriptions#0::Description": (
                "我们发现了一个补给缓存。"
            ),
            "MissionObjectiveTexts::MissionDescriptions#1::Description": ("营救VIP。"),
            "X2DownloadableContentInfo_AdditionalUnitInfo"
            "::UnitTacticalInfo#0::UnitDescription": ("外星战士部族成员。"),
            "X2DownloadableContentInfo_WotCBallisticShields"
            "::m_arrAnimationPoses#0::AnimationDisplayName": ("盾剑 1"),
            "Rend X2AbilityTemplate::LocFriendlyName": "撕裂",
            "Rend X2AbilityTemplate::LocLongDescription": (
                "撕开目标，造成<Ability:RENDDMG/>点伤害。"
            ),
        }

        target_path = tmp_path / "struct_append.chn"
        target_file = converter.build_target_file(
            source=source,
            translations=translations,
            target_lang="zh_Hans",
            target_path=target_path,
        )
        loc_writer.write(target_file, target_path)

        reparsed = parser.parse(target_path)

        # Structural invariants
        assert reparsed.entry_count == source.entry_count
        assert reparsed.lang == "zh_Hans"
        assert reparsed.encoding == "utf-16-le"
        assert len(reparsed.sections) == len(source.sections)

        # MissionDescriptions#0 translation survives round-trip
        mission_section = next(
            s for s in reparsed.sections if s.header.raw == "MissionObjectiveTexts"
        )
        first = mission_section.entries[0]
        assert first.is_append
        assert first.struct_fields is not None
        desc = next(f for f in first.struct_fields if f.key == "Description")
        assert desc.value == "我们发现了一个补给缓存。"

        # Non-translatable MissionIndex untouched
        idx = next(f for f in first.struct_fields if f.key == "MissionIndex")
        assert idx.value == "0"

        # UnitTacticalInfo round-trip, UnitName stays as identifier
        unit_section = next(
            s
            for s in reparsed.sections
            if s.header.raw == "X2DownloadableContentInfo_AdditionalUnitInfo"
        )
        unit_entry = unit_section.entries[0]
        assert unit_entry.struct_fields is not None
        name_field = next(f for f in unit_entry.struct_fields if f.key == "UnitName")
        assert name_field.value == "Muton"  # not translated
        desc_field = next(
            f for f in unit_entry.struct_fields if f.key == "UnitDescription"
        )
        assert desc_field.value == "外星战士部族成员。"

        # Placeholder inside translation survives byte-for-byte
        rend_section = next(
            s for s in reparsed.sections if s.header.raw == "Rend X2AbilityTemplate"
        )
        long_desc = next(
            e for e in rend_section.entries if e.key == "LocLongDescription"
        )
        assert "<Ability:RENDDMG/>" in long_desc.value

    def test_fallback_entries_survive_roundtrip(
        self,
        parser: LocFileParser,
        converter: CorpusConverter,
        loc_writer: LocFileWriter,
        struct_append_int: Path,
        tmp_path: Path,
    ) -> None:
        """Untranslated entries keep source values through write+re-parse."""
        source = parser.parse(struct_append_int)
        target_path = tmp_path / "out.chn"
        target_file = converter.build_target_file(
            source=source,
            translations={},
            target_lang="zh_Hans",
            target_path=target_path,
        )
        loc_writer.write(target_file, target_path)

        reparsed = parser.parse(target_path)
        assert reparsed.entry_count == source.entry_count

        # Every value matches source (no translations applied)
        for src_sec, r_sec in zip(source.sections, reparsed.sections, strict=True):
            assert src_sec.header.raw == r_sec.header.raw
            for src_e, r_e in zip(src_sec.entries, r_sec.entries, strict=True):
                assert src_e.value == r_e.value
