"""Unit tests for LocFileParser — organized by method."""

from pathlib import Path

import pytest

from src.core.parser import LocFileParser
from src.models._share import PlaceholderType, SectionHeaderFormat

parser = LocFileParser()


class TestInferEncoding:
    def test_utf16le_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"\xff\xfe" + "hello".encode("utf-16-le"))
        assert parser._infer_encoding(p) == "utf-16-le"

    def test_utf16be_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"\xfe\xff" + "hello".encode("utf-16-be"))
        assert parser._infer_encoding(p) == "utf-16-be"

    def test_utf8sig_bom_only(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"\xef\xbb\xbf")
        assert parser._infer_encoding(p) == "utf-8-sig"

    def test_utf8sig_bom_with_content(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"\xef\xbb\xbf" + b"hello")
        assert parser._infer_encoding(p) == "utf-8-sig"

    def test_utf8_no_bom(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        assert parser._infer_encoding(p) == "utf-8"

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        # 0x92 = right single quote in Windows-1252, invalid in UTF-8
        p = tmp_path / "test.bin"
        p.write_bytes(b"didn\x92t work")
        assert parser._infer_encoding(p) == "latin1"

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"")
        assert parser._infer_encoding(p) == "utf-8"

    def test_fixture_sample_int(self, sample_int: Path) -> None:
        assert parser._infer_encoding(sample_int) == "utf-16-le"


class TestInferLang:
    @pytest.mark.parametrize(
        ("ext", "expected"),
        [
            ("int", "en"),
            ("chn", "zh_Hans"),
            ("cht", "zh_Hant"),
            ("deu", "de"),
            ("esn", "es"),
            ("fra", "fr"),
            ("ita", "it"),
            ("jpn", "ja"),
            ("kor", "ko"),
            ("pol", "pl"),
            ("rus", "ru"),
        ],
    )
    def test_known_extensions(self, ext: str, expected: str) -> None:
        assert parser._infer_lang(Path(f"/fake/file.{ext}")) == expected

    def test_uppercase_extension(self) -> None:
        assert parser._infer_lang(Path("/fake/file.INT")) == "en"

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parser._infer_lang(Path("/fake/file.xyz"))

    def test_no_extension_raises(self) -> None:
        with pytest.raises(ValueError):
            parser._infer_lang(Path("/fake/noext"))


class TestReadFile:
    def test_utf16le_basic(self, make_loc_file) -> None:
        p = make_loc_file("[Section]\nKey=Val")
        lines = parser._read_file(p, "utf-16-le")
        assert "[Section]" in lines
        assert "Key=Val" in lines

    def test_bom_stripping(self, make_loc_file) -> None:
        p = make_loc_file("[Section]")
        lines = parser._read_file(p, "utf-16-le")
        assert not lines[0].startswith("\ufeff")

    def test_utf8sig_read(self, make_loc_file) -> None:
        p = make_loc_file("[Section]\nKey=Val", encoding="utf-8-sig")
        lines = parser._read_file(p, "utf-8-sig")
        assert "[Section]" in lines

    def test_unicode_content(self, make_loc_file) -> None:
        p = make_loc_file('[Section]\nKey="确定"')
        lines = parser._read_file(p, "utf-16-le")
        assert any("确定" in line for line in lines)

    def test_wrong_encoding_raises(self, make_loc_file) -> None:
        p = make_loc_file("[Section]", encoding="utf-16-le")
        with pytest.raises(UnicodeDecodeError):
            parser._read_file(p, "utf-8")


class TestParseSectionHeader:
    def test_format_a_class_only(self) -> None:
        h = parser._parse_section_header("UIUtilities_Text")
        assert h.format == SectionHeaderFormat.CLASS_ONLY
        assert h.name == "UIUtilities_Text"
        assert h.object_name is None
        assert h.class_name == "UIUtilities_Text"
        assert h.package is None

    def test_format_b_object_class(self) -> None:
        h = parser._parse_section_header("BattleScanner X2AbilityTemplate")
        assert h.format == SectionHeaderFormat.OBJECT_CLASS
        assert h.name == "BattleScanner"
        assert h.object_name == "BattleScanner"
        assert h.class_name == "X2AbilityTemplate"
        assert h.package is None

    def test_format_c_package_class(self) -> None:
        h = parser._parse_section_header("XComGame.UIFinalShell")
        assert h.format == SectionHeaderFormat.PACKAGE_CLASS
        assert h.name == "XComGame.UIFinalShell"
        assert h.class_name == "UIFinalShell"
        assert h.package == "XComGame"
        assert h.object_name is None

    def test_format_d_archetype_class(self) -> None:
        h = parser._parse_section_header(
            "Archetypes.ARC_AdventSecureDoor_x1 XComInteractiveLevelActor"
        )
        assert h.format == SectionHeaderFormat.ARCHETYPE_CLASS
        assert h.name == "ARC_AdventSecureDoor_x1"
        assert h.object_name == "ARC_AdventSecureDoor_x1"
        assert h.class_name == "XComInteractiveLevelActor"
        assert h.package == "Archetypes"

    def test_format_c_nested_dots(self) -> None:
        h = parser._parse_section_header("A.B.C")
        assert h.format == SectionHeaderFormat.PACKAGE_CLASS
        assert h.package == "A.B"
        assert h.class_name == "C"

    def test_format_d_nested_dots(self) -> None:
        h = parser._parse_section_header("A.B.MyObj SomeClass")
        assert h.format == SectionHeaderFormat.ARCHETYPE_CLASS
        assert h.package == "A.B"
        assert h.object_name == "MyObj"
        assert h.class_name == "SomeClass"

    def test_whitespace_stripped(self) -> None:
        h = parser._parse_section_header("  UIUtilities_Text  ")
        assert h.format == SectionHeaderFormat.CLASS_ONLY
        assert h.name == "UIUtilities_Text"

    def test_raw_preserved(self) -> None:
        h = parser._parse_section_header("XComGame.UIFinalShell")
        assert h.raw == "XComGame.UIFinalShell"


class TestExtractPlaceholders:
    def test_xgparam(self) -> None:
        phs = parser._extract_placeholders("<XGParam:IntValue0/>")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.XGPARAM
        assert phs[0].pattern == "<XGParam:IntValue0/>"

    def test_xgparam_with_label(self) -> None:
        phs = parser._extract_placeholders("<XGParam:StrValue0/!RegionName/>")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.XGPARAM

    def test_ability(self) -> None:
        phs = parser._extract_placeholders(
            "gains +<Ability:LIGHTNINGSTRIKEMOVEBONUS/> mobility"
        )
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.ABILITY

    def test_bullet(self) -> None:
        phs = parser._extract_placeholders("<Bullet/> text")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.BULLET

    def test_heal(self) -> None:
        phs = parser._extract_placeholders("Heals <Heal/> HP")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.HEAL

    def test_br(self) -> None:
        phs = parser._extract_placeholders("line1<br/>line2")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.BR

    def test_xml_var(self) -> None:
        phs = parser._extract_placeholders("Photo by <Photobooth:FirstName0/>")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.XML_VAR
        assert phs[0].pattern == "<Photobooth:FirstName0/>"

    def test_xml_self_close(self) -> None:
        phs = parser._extract_placeholders("Focus: <FocusAmount/>")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.XML_SELF_CLOSE
        assert phs[0].pattern == "<FocusAmount/>"

    def test_html_font_pair(self) -> None:
        phs = parser._extract_placeholders("<font color='#FF0000'>red</font> normal")
        assert len(phs) == 2
        assert all(p.type == PlaceholderType.HTML for p in phs)

    def test_html_headings(self) -> None:
        phs = parser._extract_placeholders("<h2>Credits</h2>")
        assert len(phs) == 2
        assert all(p.type == PlaceholderType.HTML for p in phs)

    def test_html_br_no_slash(self) -> None:
        phs = parser._extract_placeholders("line1<br>line2")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.HTML

    def test_percent_wrapped(self) -> None:
        phs = parser._extract_placeholders("replace %modnames% here")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.PERCENT_WRAPPED
        assert phs[0].pattern == "%modnames%"

    def test_percent_single(self) -> None:
        phs = parser._extract_placeholders("%A BEGIN")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.PERCENT
        assert phs[0].pattern == "%A"

    def test_newline(self) -> None:
        phs = parser._extract_placeholders("line1\\nline2")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.NEWLINE

    def test_no_placeholders(self) -> None:
        assert parser._extract_placeholders("Just normal text.") == []

    def test_empty_string(self) -> None:
        assert parser._extract_placeholders("") == []

    def test_mixed_sorted_by_span(self) -> None:
        phs = parser._extract_placeholders(
            "<Bullet/> Deals <XGParam:IntValue0/> damage.<br/>"
        )
        assert len(phs) == 3
        assert phs[0].type == PlaceholderType.BULLET
        assert phs[1].type == PlaceholderType.XGPARAM
        assert phs[2].type == PlaceholderType.BR
        assert phs[0].span[0] < phs[1].span[0] < phs[2].span[0]

    def test_span_accuracy(self) -> None:
        phs = parser._extract_placeholders("abc<Bullet/>def")
        assert phs[0].span == (3, 12)

    def test_priority_percent_wrapped_over_percent(self) -> None:
        """PERCENT_WRAPPED should win over PERCENT for %AbilityName%."""
        phs = parser._extract_placeholders("%AbilityName%")
        assert len(phs) == 1
        assert phs[0].type == PlaceholderType.PERCENT_WRAPPED


class TestParseEntry:
    def test_simple_quoted(self) -> None:
        e = parser._parse_entry(("", "m_strOK", '"OK"'), 1, [])
        assert e.key == "m_strOK"
        assert e.value == "OK"
        assert e.raw_value == '"OK"'
        assert not e.is_array
        assert not e.is_append

    def test_unquoted_value(self) -> None:
        e = parser._parse_entry(("", "CharStatLabels[eStat_HP]", "HP"), 5, [])
        assert e.value == "HP"

    def test_append_syntax(self) -> None:
        e = parser._parse_entry(("+", "Key", '"Appended"'), 1, [])
        assert e.is_append

    def test_numeric_array_index(self) -> None:
        e = parser._parse_entry(("", "RankNames[0]", '"Rookie"'), 10, [])
        assert e.is_array
        assert e.array_index == 0
        assert e.key == "RankNames[0]"
        assert e.value == "Rookie"

    def test_enum_array_index(self) -> None:
        e = parser._parse_entry(("", "CharStatLabels[eStat_HP]", "HP"), 5, [])
        assert e.is_array
        assert e.array_index is None  # enum, not numeric

    def test_no_array(self) -> None:
        e = parser._parse_entry(("", "SimpleKey", '"val"'), 1, [])
        assert not e.is_array
        assert e.array_index is None

    def test_append_struct(self) -> None:
        raw = (
            '(MissionFamily="Recover_LW", '
            'Description="Found <XGParam:StrValue0/!Region/>", '
            "MissionIndex=0)"
        )
        e = parser._parse_entry(("+", "MissionDescriptions", raw), 20, [])
        assert e.is_append
        assert e.struct_fields is not None
        assert len(e.struct_fields) == 3

        desc = e.struct_fields[1]
        assert desc.key == "Description"
        assert len(desc.placeholders) == 1
        assert desc.placeholders[0].type == PlaceholderType.XGPARAM

        idx = e.struct_fields[2]
        assert idx.key == "MissionIndex"
        assert idx.value == "0"
        assert idx.placeholders == []

    def test_append_non_struct(self) -> None:
        e = parser._parse_entry(("+", "Key", '"plain value"'), 1, [])
        assert e.is_append
        assert e.struct_fields is None

    def test_comments_attached(self) -> None:
        e = parser._parse_entry(("", "key", '"val"'), 3, ["c1", "c2"])
        assert e.comments == ["c1", "c2"]

    def test_line_number(self) -> None:
        e = parser._parse_entry(("", "key", '"val"'), 42, [])
        assert e.line_number == 42

    def test_empty_value(self) -> None:
        e = parser._parse_entry(("", "key", ""), 1, [])
        assert e.value == ""
        assert e.raw_value == ""

    def test_empty_quoted_value(self) -> None:
        e = parser._parse_entry(("", "key", '""'), 1, [])
        assert e.value == ""

    def test_unclosed_string_literal_strips_leading_quote(self) -> None:
        """Malformed .int files with missing closing `"` — observed in
        real mods (e.g. RealModFiles/2867288932 T2/T3/T4 weapon templates).
        The parser must strip the stray leading `"` so it doesn't leak
        into Weblate uploads or glossary exports.
        """
        e = parser._parse_entry(("", "AbilityDescName", "\"Bala'Kal Cannon"), 1, [])
        assert e.value == "Bala'Kal Cannon"
        assert e.raw_value == "\"Bala'Kal Cannon"

    def test_unopened_string_literal_strips_trailing_quote(self) -> None:
        """Mirror case: trailing `"` without an opening one. Strip the
        orphan rather than letting it pollute downstream views.
        """
        e = parser._parse_entry(("", "Key", 'Psionic Amplifiers"'), 1, [])
        assert e.value == "Psionic Amplifiers"
        assert e.raw_value == 'Psionic Amplifiers"'

    def test_lone_quote_strips_to_empty(self) -> None:
        """Single `"` (author typo): strip to empty string."""
        e = parser._parse_entry(("", "Key", '"'), 1, [])
        assert e.value == ""

    def test_inline_comment_stripped(self) -> None:
        """UE3 `.int` files support trailing `;comment` on entry lines:

            Key="Lifetime Stats"    ;gets appended with [unit name]

        The regex captures the entire rest of the line including the
        comment. Parser must strip the comment before doing quote
        extraction, otherwise a valid quoted value gets misclassified
        as an unclosed string literal and leaks the stray `"`.
        """
        e = parser._parse_entry(
            ("", "Key", '"Lifetime Stats"    ;gets appended with [unit name]'),
            1,
            [],
        )
        assert e.value == "Lifetime Stats"
        assert e.raw_value == '"Lifetime Stats"'

    def test_inline_comment_preserves_semicolon_inside_quotes(self) -> None:
        """A `;` inside a quoted string is legal content, NOT a comment."""
        e = parser._parse_entry(("", "Key", '"a;b;c"'), 1, [])
        assert e.value == "a;b;c"

    def test_inline_comment_with_escaped_quote(self) -> None:
        """`\\"` inside the quoted section must not flip the state
        machine and trigger a false comment cut.
        """
        e = parser._parse_entry(
            ("", "Key", '"He said \\"hi;lo\\""  ;trailing note'), 1, []
        )
        assert e.raw_value == '"He said \\"hi;lo\\""'
        assert e.value == 'He said \\"hi;lo\\"'

    def test_inline_comment_on_unquoted_value(self) -> None:
        """Comments can also trail unquoted values."""
        e = parser._parse_entry(("", "Key", "42   ;meaningful constant"), 1, [])
        assert e.value == "42"

    def test_double_opening_quote_pattern(self) -> None:
        # Real-mod case: `Key=""Cover Me"` — two opening quotes and one
        # closing. Strict outer strip produces `"Cover Me` which then
        # still needs the residual leading `"` cleaned up.
        e = parser._parse_entry(("", "LocFlyOverText", '""Cover Me"'), 1, [])
        assert e.value == "Cover Me"

    def test_triple_quote_becomes_empty(self) -> None:
        # Real-mod case: three consecutive quotes as the entire value.
        # Author probably meant an empty string but typed an extra quote.
        # After outer strip the value is a lone quote; edge cleanup drops it.
        e = parser._parse_entry(("", "TacticalText", '"""'), 1, [])
        assert e.value == ""

    def test_trailing_double_quote_pattern(self) -> None:
        # Mirror of double-opening: `Key="Value""` — open, value, two
        # closing. Strict strip produces `Value"`; edge cleanup drops
        # the trailing residual.
        e = parser._parse_entry(("", "Key", '"Value""'), 1, [])
        assert e.value == "Value"

    def test_escaped_quote_not_stripped_by_edge_cleanup(self) -> None:
        # Critical correctness: `"\"Hello\""` strips to `\"Hello\"`,
        # which ends with `\"` (a legitimate escape). The edge cleanup
        # must NOT strip that trailing quote — only unescaped strays.
        e = parser._parse_entry(("", "Key", '"\\"Hello\\""'), 1, [])
        assert e.value == '\\"Hello\\"'

    def test_middle_stray_quote_upgraded_to_escape(self) -> None:
        # Real-mod case (Bug 4) from RealModFiles/1123582370 line 16:
        #   LocLongDescription="A powerful form... any single action." 3 turn cooldown."
        # Three quotes total. The strict strip takes [1:-1], producing
        #   A powerful form... any single action." 3 turn cooldown.
        # The middle stray `"` must be upgraded to `\"` so writeback
        # produces a valid escaped string that round-trips cleanly.
        raw = (
            '"A powerful form of Overwatch. Instead of firing '
            'automatically, perform any single action." 3 turn cooldown."'
        )
        e = parser._parse_entry(("", "LocLongDescription", raw), 1, [])
        expected = (
            "A powerful form of Overwatch. Instead of firing "
            'automatically, perform any single action.\\" 3 turn cooldown.'
        )
        assert e.value == expected

    def test_legitimate_escape_not_double_upgraded(self) -> None:
        # Already-escaped `\"` must pass through untouched — the
        # previously-emitted `\` blocks the upgrade.
        e = parser._parse_entry(("", "Key", '"She said \\"hi\\" again"'), 1, [])
        assert e.value == 'She said \\"hi\\" again'

    def test_multiple_middle_strays_all_upgraded(self) -> None:
        # Multiple stray quotes in a single value — each gets upgraded
        # independently.
        e = parser._parse_entry(("", "Key", '"a"b"c"d"'), 1, [])
        # Strict strip: a"b"c"d → Bug 4 upgrade: a\"b\"c\"d
        assert e.value == 'a\\"b\\"c\\"d'

    def test_mixed_legitimate_and_stray_upgrades_only_strays(self) -> None:
        # Input: `"already \"clean\" and "dirty" mixed"`
        # Strict strip: `already \"clean\" and "dirty" mixed`
        # Bug 4: upgrades the two stray `"` around "dirty" but leaves
        # the `\"clean\"` escapes intact.
        raw = '"already \\"clean\\" and "dirty" mixed"'
        e = parser._parse_entry(("", "Key", raw), 1, [])
        assert e.value == 'already \\"clean\\" and \\"dirty\\" mixed'


class TestParseStructFields:
    def test_mixed_fields(self) -> None:
        fields = parser._parse_struct_fields(
            'MissionFamily="Recover_LW", Description="We found it", MissionIndex=0'
        )
        assert len(fields) == 3
        assert fields[0].key == "MissionFamily"
        assert fields[0].value == "Recover_LW"
        assert fields[1].key == "Description"
        assert fields[1].value == "We found it"
        assert fields[2].key == "MissionIndex"
        assert fields[2].value == "0"

    def test_comma_in_quotes(self) -> None:
        fields = parser._parse_struct_fields('Key1="hello, world", Key2=42')
        assert len(fields) == 2
        assert fields[0].value == "hello, world"
        assert fields[1].value == "42"

    def test_numeric_no_placeholders(self) -> None:
        fields = parser._parse_struct_fields("Count=5")
        assert len(fields) == 1
        assert fields[0].placeholders == []

    def test_quoted_with_placeholder(self) -> None:
        fields = parser._parse_struct_fields(
            'Description="Region <XGParam:StrValue0/>"'
        )
        assert len(fields) == 1
        assert len(fields[0].placeholders) == 1

    def test_no_equals_skipped(self) -> None:
        fields = parser._parse_struct_fields("BadField")
        assert fields == []

    def test_multiple_equals_split_on_first(self) -> None:
        fields = parser._parse_struct_fields('Key="a=b=c"')
        assert len(fields) == 1
        assert fields[0].value == "a=b=c"

    def test_empty_content(self) -> None:
        fields = parser._parse_struct_fields("")
        assert fields == []


class TestScanLines:
    def test_empty_lines(self) -> None:
        sections, header_comments = parser._scan_lines([], Path("/fake"))
        assert sections == []
        assert header_comments == []

    def test_single_section_single_entry(self) -> None:
        lines = ["[Section]", 'Key="Val"']
        sections, _ = parser._scan_lines(lines, Path("/fake"))
        assert len(sections) == 1
        assert len(sections[0].entries) == 1

    def test_multiple_sections(self) -> None:
        lines = ["[S1]", 'K1="V1"', "[S2]", 'K2="V2"']
        sections, _ = parser._scan_lines(lines, Path("/fake"))
        assert len(sections) == 2

    def test_header_comments(self) -> None:
        lines = ["; header comment", "[Section]", 'Key="Val"']
        sections, header_comments = parser._scan_lines(lines, Path("/fake"))
        assert header_comments == ["header comment"]
        assert sections[0].comments == []

    def test_section_comments(self) -> None:
        lines = ["[S1]", 'K="V"', "; between", "[S2]", 'K2="V2"']
        sections, _ = parser._scan_lines(lines, Path("/fake"))
        assert sections[1].comments == ["between"]

    def test_entry_comments(self) -> None:
        lines = ["[S]", "; entry comment", 'K="V"']
        sections, _ = parser._scan_lines(lines, Path("/fake"))
        assert sections[0].entries[0].comments == ["entry comment"]

    def test_entry_before_section_discarded(self) -> None:
        """UE3 engine silently discards entries before any section header."""
        lines = ['Key="Val"', "[S]", 'K2="V2"']
        sections, _ = parser._scan_lines(lines, Path("/fake"))
        assert len(sections) == 1
        assert len(sections[0].entries) == 1
        assert sections[0].entries[0].key == "K2"

    def test_blank_and_bom_skipped(self) -> None:
        lines = ["\ufeff", "", "  ", "[S]", 'K="V"']
        sections, _ = parser._scan_lines(lines, Path("/fake"))
        assert len(sections) == 1
        assert len(sections[0].entries) == 1
