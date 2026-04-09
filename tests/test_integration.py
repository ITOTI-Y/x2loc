"""Integration tests for LocFileParser.parse() and BilingualAligner.align()."""

from pathlib import Path

import pytest

from src.core.aligner import BilingualAligner
from src.core.extractor import TermExtractor
from src.core.parser import LocFileParser
from src.models._share import PlaceholderType, SectionHeaderFormat


class TestParseIntegration:
    def test_sample_int(self, parser: LocFileParser, sample_int: Path) -> None:
        result = parser.parse(sample_int)

        assert result.lang == "en"
        assert result.encoding == "utf-16-le"
        assert result.path.is_absolute()
        assert len(result.sections) == 9

        # Header comments
        assert len(result.header_comments) == 2
        assert "File header comment" in result.header_comments[0]
        assert "Another header comment" in result.header_comments[1]

        # Format A: CLASS_ONLY
        sec0 = result.sections[0]
        assert sec0.header.format == SectionHeaderFormat.CLASS_ONLY
        assert sec0.header.name == "UIUtilities_Text"
        assert len(sec0.entries) == 2

        # Format B: OBJECT_CLASS
        sec1 = result.sections[1]
        assert sec1.header.format == SectionHeaderFormat.OBJECT_CLASS
        assert sec1.header.object_name == "BattleScanner"
        assert sec1.header.class_name == "X2AbilityTemplate"

        # Format C: PACKAGE_CLASS
        sec2 = result.sections[2]
        assert sec2.header.format == SectionHeaderFormat.PACKAGE_CLASS
        assert sec2.header.package == "XComGame"
        assert sec2.header.class_name == "UIFinalShell"

        # Format D: ARCHETYPE_CLASS
        sec3 = result.sections[3]
        assert sec3.header.format == SectionHeaderFormat.ARCHETYPE_CLASS
        assert sec3.header.package == "Archetypes"
        assert sec3.header.object_name == "ARC_AdventSecureDoor_x1"

        # Array entries (X2ExperienceConfig)
        sec4 = result.sections[4]
        assert sec4.entries[0].is_array
        assert sec4.entries[0].array_index == 0
        assert sec4.entries[0].value == "Rookie"
        # Enum array index
        assert sec4.entries[3].is_array
        assert sec4.entries[3].array_index is None
        assert sec4.entries[3].value == "HP"

        # Entry with comment (X2TacticalGameRulesetDataStructures)
        sec5 = result.sections[5]
        assert sec5.entries[0].comments == ["Stat labels"]

        # Append struct (MissionObjectiveTexts)
        sec6 = result.sections[6]
        append_entry = sec6.entries[0]
        assert append_entry.is_append
        assert append_entry.struct_fields is not None
        assert len(append_entry.struct_fields) == 3

        # Percent placeholder (UIFinalShell)
        sec7 = result.sections[7]
        btn = sec7.entries[0]
        assert any(p.type == PlaceholderType.PERCENT for p in btn.placeholders)
        assert btn.comments == ["LWOTC Needs Translation"]

        # entry_count computed field
        assert result.entry_count == sum(len(s.entries) for s in result.sections)
        assert result.entry_count > 0

    def test_sample_chn(self, parser: LocFileParser, sample_chn: Path) -> None:
        result = parser.parse(sample_chn)
        assert result.lang == "zh_Hans"
        assert len(result.sections) == 2
        assert result.sections[0].entries[0].value == "确定"

    def test_empty_file(self, parser: LocFileParser, empty_int: Path) -> None:
        result = parser.parse(empty_int)
        assert result.sections == []
        assert result.entry_count == 0

    def test_comments_only(
        self, parser: LocFileParser, comments_only_int: Path
    ) -> None:
        result = parser.parse(comments_only_int)
        assert result.sections == []
        # Comments before any section become header_comments
        assert len(result.header_comments) == 0  # no section → never flushed

    def test_entry_before_section_discarded(
        self, parser: LocFileParser, entry_before_section_int: Path
    ) -> None:
        """UE3 engine silently discards entries before any section header."""
        result = parser.parse(entry_before_section_int)
        # fixture: Key="Value" \n [Section] \n Other="data"
        assert len(result.sections) == 1
        assert result.sections[0].entries[0].key == "Other"
        assert result.entry_count == 1

    def test_lang_override(self, parser: LocFileParser, sample_int: Path) -> None:
        result = parser.parse(sample_int, lang_override="ja")
        assert result.lang == "ja"

    def test_lang_override_none_uses_extension(
        self, parser: LocFileParser, sample_int: Path
    ) -> None:
        result = parser.parse(sample_int, lang_override=None)
        assert result.lang == "en"

    def test_file_not_found(self, parser: LocFileParser) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.int"))

    def test_path_is_absolute(self, parser: LocFileParser, sample_int: Path) -> None:
        result = parser.parse(sample_int)
        assert result.path.is_absolute()

    def test_placeholder_showcase(
        self, parser: LocFileParser, sample_int: Path
    ) -> None:
        """Verify all 11 placeholder types appear in PlaceholderShowcase section."""
        result = parser.parse(sample_int)
        showcase = result.sections[8]
        assert showcase.header.name == "PlaceholderShowcase"

        all_types = {p.type for e in showcase.entries for p in e.placeholders}
        expected = {
            PlaceholderType.XGPARAM,
            PlaceholderType.ABILITY,
            PlaceholderType.XML_VAR,
            PlaceholderType.BULLET,
            PlaceholderType.HEAL,
            PlaceholderType.BR,
            PlaceholderType.XML_SELF_CLOSE,
            PlaceholderType.HTML,
            PlaceholderType.PERCENT_WRAPPED,
            PlaceholderType.PERCENT,
            PlaceholderType.NEWLINE,
        }
        assert all_types == expected


class TestParseWithDynamicFiles:
    def test_utf8sig_file(self, parser: LocFileParser, make_loc_file) -> None:
        p = make_loc_file(
            '[Section]\nKey="Value"',
            filename="test.int",
            encoding="utf-8-sig",
        )
        result = parser.parse(p)
        assert result.encoding == "utf-8-sig"
        assert result.sections[0].entries[0].value == "Value"

    def test_utf8_plain_file(self, parser: LocFileParser, make_loc_file) -> None:
        p = make_loc_file(
            '[Section]\nKey="Value"',
            filename="test.int",
            encoding="utf-8",
        )
        result = parser.parse(p)
        assert result.encoding == "utf-8"

    def test_utf16be_file(self, parser: LocFileParser, make_loc_file) -> None:
        p = make_loc_file(
            '[Section]\nKey="Value"',
            filename="test.int",
            encoding="utf-16-be",
        )
        result = parser.parse(p)
        assert result.encoding == "utf-16-be"

    def test_unknown_extension_raises(
        self, parser: LocFileParser, make_loc_file
    ) -> None:
        p = make_loc_file('[Section]\nKey="Value"', filename="test.xyz")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parser.parse(p)

    def test_unicode_values_preserved(
        self, parser: LocFileParser, make_loc_file
    ) -> None:
        p = make_loc_file('[Section]\nKey="确定取消"')
        result = parser.parse(p)
        assert result.sections[0].entries[0].value == "确定取消"


class TestAlignIntegration:
    def test_align_fixture_files(
        self,
        parser: LocFileParser,
        aligner: BilingualAligner,
        align_source_int: Path,
        align_target_chn: Path,
    ) -> None:
        src = parser.parse(align_source_int)
        tgt = parser.parse(align_target_chn)
        corpus = aligner.align(src, tgt)

        # Source: 3 UIUtilities + 2 BattleScanner + 3 appends = 8
        # Target: 2 UIUtilities + 3 BattleScanner + 2 appends = 7
        # Aligned: OK, Cancel, FriendlyName, HelpText, Desc#0, Desc#1 = 6
        assert corpus.aligned_count == 6

        # Source only: m_strSourceOnly, MissionDescriptions#2
        assert len(corpus.source_only) == 2
        assert "UIUtilities_Text::m_strSourceOnly" in corpus.source_only
        assert "MissionObjectiveTexts::MissionDescriptions#2" in corpus.source_only

        # Target only: LocTargetOnly
        assert len(corpus.target_only) == 1
        assert "BattleScanner X2AbilityTemplate::LocTargetOnly" in corpus.target_only

        # Total entries: 8 source + 1 target_only = 9
        assert len(corpus.entries) == 9

    def test_align_source_only_corpus(
        self,
        parser: LocFileParser,
        aligner: BilingualAligner,
        align_source_int: Path,
    ) -> None:
        src = parser.parse(align_source_int)
        corpus = aligner.align(src, target_lang="ja")

        assert corpus.target_lang == "ja"
        assert corpus.target_path is None
        assert corpus.aligned_count == 0
        assert len(corpus.source_only) == 8
        assert corpus.target_only == []


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

skip_no_data = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="data/ directory not available"
)


@skip_no_data
class TestRealData:
    def test_xcomgame_int(self, parser: LocFileParser) -> None:
        p = DATA_DIR / "GameExample" / "INT" / "XComGame.int"
        if not p.exists():
            pytest.skip("XComGame.int not found")
        result = parser.parse(p)
        assert result.lang == "en"
        assert result.encoding == "utf-16-le"
        assert result.entry_count > 1000
        assert len(result.sections) > 50

    def test_xcomgame_chn(self, parser: LocFileParser) -> None:
        p = DATA_DIR / "GameExample" / "CHN" / "XComGame.chn"
        if not p.exists():
            pytest.skip("XComGame.chn not found")
        result = parser.parse(p)
        assert result.lang == "zh_Hans"
        assert result.entry_count > 1000

    def test_align_xcomgame(
        self, parser: LocFileParser, aligner: BilingualAligner
    ) -> None:
        int_path = DATA_DIR / "GameExample" / "INT" / "XComGame.int"
        chn_path = DATA_DIR / "GameExample" / "CHN" / "XComGame.chn"

        if not int_path.exists() or not chn_path.exists():
            pytest.skip("XComGame .int/.chn pair not found")

        src = parser.parse(int_path)
        tgt = parser.parse(chn_path)
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count > 1000
        assert len(corpus.entries) > 0

    def test_extract_xcomgame(
        self,
        parser: LocFileParser,
        aligner: BilingualAligner,
        extractor: TermExtractor,
    ) -> None:
        int_path = DATA_DIR / "GameExample" / "INT" / "XComGame.int"
        chn_path = DATA_DIR / "GameExample" / "CHN" / "XComGame.chn"

        if not int_path.exists() or not chn_path.exists():
            pytest.skip("XComGame .int/.chn pair not found")

        src = parser.parse(int_path)
        tgt = parser.parse(chn_path)
        corpus = aligner.align(src, tgt)
        glossary = extractor.extract([corpus])

        assert glossary.term_count > 500

        ability_terms = [t for t in glossary.terms if t.category == "ability"]
        assert len(ability_terms) > 100

        ph_terms = [t for t in glossary.terms if t.do_not_translate]
        assert len(ph_terms) > 5

        # Verify no empty source values
        for term in glossary.terms:
            if not term.do_not_translate:
                assert term.source != ""
