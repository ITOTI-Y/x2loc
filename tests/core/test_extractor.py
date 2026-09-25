"""Unit tests for TermExtractor."""

from pathlib import Path

from src.core.extractor import (
    NAME_KEY_HINTS,
    TEMPLATE_RULES,
    TermExtractor,
)
from src.models._share import PlaceholderType, SectionHeaderFormat
from src.models.corpus import BilingualCorpus, BilingualEntry
from src.models.entry import EntrySchema, PlaceholderSchema
from src.models.section import SectionHeader


def _entry(
    key: str,
    value: str,
    is_append: bool = False,
    struct_fields: list | None = None,
    placeholders: list[PlaceholderSchema] | None = None,
) -> EntrySchema:
    return EntrySchema(
        key=key,
        raw_value=f'"{value}"',
        value=value,
        is_append=is_append,
        struct_fields=struct_fields,
        placeholders=placeholders or [],
        line_number=1,
    )


def _header(
    raw: str,
    class_name: str | None = None,
    fmt: SectionHeaderFormat = SectionHeaderFormat.CLASS_ONLY,
) -> SectionHeader:
    return SectionHeader(
        raw=raw,
        format=fmt,
        name=raw,
        class_name=class_name or raw,
    )


def _bilingual(
    compound_key: str,
    source: EntrySchema,
    target: EntrySchema | None,
    header: SectionHeader,
) -> BilingualEntry:
    return BilingualEntry(
        compound_key=compound_key,
        source=source,
        target=target,
        section_header=header,
    )


def _corpus(
    entries: list[BilingualEntry],
    source_lang: str = "en",
    target_lang: str = "zh_Hans",
) -> BilingualCorpus:
    return BilingualCorpus(
        source_lang=source_lang,
        target_lang=target_lang,
        source_path=Path("/src/test.int"),
        target_path=Path("/tgt/test.chn"),
        entries=entries,
        namespace="test-extractor",
        mod_title="Test Extractor Fixture",
    )


class TestClassify:
    """Tests for TermExtractor._classify()."""

    def test_rule_a_ability(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Rend X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocFriendlyName", "Rend")
        tgt = _entry("LocFriendlyName", "撕裂")
        entry = _bilingual("Rend X2AbilityTemplate::LocFriendlyName", src, tgt, hdr)

        result = extractor._classify(entry)
        assert result == ("ability", 0)

    def test_rule_a_weapon(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "AssaultRifle_CV X2WeaponTemplate",
            class_name="X2WeaponTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("FriendlyName", "Assault Rifle")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("weapon", 0)

    def test_rule_a_soldier_class(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Ranger X2SoldierClassTemplate",
            class_name="X2SoldierClassTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("DisplayName", "Ranger")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("soldier_class", 0)

    def test_rule_a_cosmetic(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Hair_A X2BodyPartTemplate",
            class_name="X2BodyPartTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("DisplayName", "Hair Style A")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("cosmetic", 0)

    def test_rule_a_wrong_key_no_match(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Rend X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocHelpText", "Some help text for the ability")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        # Falls through to Rule C (short text check)
        assert result is None or result[1] > 0

    def test_rule_a_skip_append(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Rend X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocFriendlyName", "Rend", is_append=True)
        entry = _bilingual("key", src, None, hdr)

        # is_append=True but struct_fields is None → not excluded by global filter
        # but Rule A checks `not source.is_append` → Rule A skips
        result = extractor._classify(entry)
        # Falls through to Rule C: "Rend" is 1 word → short_text
        assert result == ("short_text", 2)

    def test_rule_a_all_template_types(self, extractor: TermExtractor) -> None:
        for class_name, (target_key, expected_cat) in TEMPLATE_RULES.items():
            hdr = _header(
                f"Obj {class_name}",
                class_name=class_name,
                fmt=SectionHeaderFormat.OBJECT_CLASS,
            )
            src = _entry(target_key, "Test Value")
            entry = _bilingual("key", src, None, hdr)

            result = extractor._classify(entry)
            assert result is not None, f"Failed for {class_name}"
            assert result[0] == expected_cat, f"Wrong category for {class_name}"
            assert result[1] == 0, f"Wrong priority for {class_name}"

    def test_rule_b_ui_generic(self, extractor: TermExtractor) -> None:
        hdr = _header("UIUtilities_Text")
        src = _entry("m_strGenericOK", "OK")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("ui_generic", 1)

    def test_rule_b_stat(self, extractor: TermExtractor) -> None:
        hdr = _header("X2TacticalGameRulesetDataStructures")
        src = _entry("CharStatLabels[eStat_HP]", "HP")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("stat", 1)

    def test_rule_b_rank(self, extractor: TermExtractor) -> None:
        hdr = _header("X2ExperienceConfig")
        src = _entry("RankNames[0]", "Rookie")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("rank", 1)

    def test_rule_b_difficulty(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        src = _entry("m_arrDifficultyTypeStrings[0]", "Easy")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("difficulty", 1)

    def test_rule_b_all_patterns(self, extractor: TermExtractor) -> None:
        test_keys = {
            "m_strGenericCancel": "ui_generic",
            "m_strDefaultHelp_Nav": "ui_nav",
            "CharStatLabels[eStat_Offense]": "stat",
            "RankNames[3]": "rank",
            "ShortNames[1]": "rank_short",
            "PsiRankNames[2]": "psi_rank",
            "m_MedalTypes[0]": "medal",
            "m_arrDifficultyTypeStrings[1]": "difficulty",
        }
        for key, expected_cat in test_keys.items():
            hdr = _header("Section")
            src = _entry(key, "Value")
            entry = _bilingual("key", src, None, hdr)

            result = extractor._classify(entry)
            assert result is not None, f"Failed for {key}"
            assert result[0] == expected_cat, f"Wrong category for {key}"
            assert result[1] == 1, f"Wrong priority for {key}"

    def test_rule_c_short_text(self, extractor: TermExtractor) -> None:
        hdr = _header("SomeSection")
        src = _entry("SomeKey", "Battle Scanner")  # 2 words
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("short_text", 2)

    def test_rule_c_three_words_boundary(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        src = _entry("Key", "one two three")  # 3 words → included
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("short_text", 2)

    def test_rule_c_four_words_without_hint_rejected(
        self, extractor: TermExtractor
    ) -> None:
        hdr = _header("Section")
        src = _entry("SomeKey", "one two three four")  # 4 words, no hint
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result is None

    def test_rule_c_four_words_with_name_hint(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        src = _entry("DisplayName", "one two three four")  # 4 words + "Name"
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("short_text", 2)

    def test_rule_c_five_words_with_hint(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        src = _entry("LocTitle", "one two three four five")  # 5 words + "Title"
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("short_text", 2)

    def test_rule_c_six_words_with_hint_rejected(
        self, extractor: TermExtractor
    ) -> None:
        hdr = _header("Section")
        src = _entry("LocTitle", "one two three four five six")  # 6 words + hint
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result is None

    def test_rule_c_all_name_hints(self, extractor: TermExtractor) -> None:
        for hint in NAME_KEY_HINTS:
            key = f"Loc{hint.capitalize()}"
            hdr = _header("Section")
            src = _entry(key, "one two three four")  # 4 words + hint
            entry = _bilingual("key", src, None, hdr)

            result = extractor._classify(entry)
            assert result is not None, f"Failed for hint {hint}"
            assert result == ("short_text", 2)

    def test_rule_c_skip_with_placeholders(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        ph = PlaceholderSchema(pattern="%A", type=PlaceholderType.PERCENT, span=(0, 2))
        src = _entry("Key", "%A BEGIN HACK", placeholders=[ph])
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result is None

    def test_rule_c_skip_sentence_ending(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        for ending in ["A trap.", "Watch out!", "Really?"]:
            src = _entry("Key", ending)
            entry = _bilingual("key", src, None, hdr)

            result = extractor._classify(entry)
            assert result is None, f"Should reject sentence: {ending}"

    def test_empty_value_rejected(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        src = _entry("Key", "")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result is None

    def test_append_struct_rejected(self, extractor: TermExtractor) -> None:
        from src.models.entry import StructFieldSchema

        hdr = _header("Section")
        fields = [StructFieldSchema(key="F", raw_value='"v"', value="v")]
        src = _entry("Key", "(F=v)", is_append=True, struct_fields=fields)
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result is None

    def test_rule_a_takes_priority_over_b(self, extractor: TermExtractor) -> None:
        """When class_name matches Rule A, Rule B should not be reached."""
        hdr = _header(
            "Obj X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocFriendlyName", "Overwatch")
        entry = _bilingual("key", src, None, hdr)

        result = extractor._classify(entry)
        assert result == ("ability", 0)


class TestExtract:
    """Tests for TermExtractor.extract()."""

    def test_single_corpus(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Aid X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        entries = [
            _bilingual(
                "key1",
                _entry("LocFriendlyName", "Aid Protocol"),
                _entry("LocFriendlyName", "援助协议"),
                hdr,
            ),
        ]
        corpus = _corpus(entries)
        glossary = extractor.extract([corpus])

        assert glossary.term_count == 1
        assert glossary.terms[0].source == "Aid Protocol"
        assert glossary.terms[0].target == "援助协议"
        assert glossary.terms[0].category == "ability"
        assert glossary.terms[0].same_as_source is False

    def test_multiple_corpora_priority(self, extractor: TermExtractor) -> None:
        """Earlier corpus has higher priority; contexts merge."""
        hdr = _header(
            "Aid X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        entry1 = _bilingual(
            "key1",
            _entry("LocFriendlyName", "Aid Protocol"),
            _entry("LocFriendlyName", "援助协议"),
            hdr,
        )
        entry2 = _bilingual(
            "key2",
            _entry("LocFriendlyName", "Aid Protocol"),
            _entry("LocFriendlyName", "援助协议"),
            hdr,
        )
        corpus1 = _corpus([entry1])
        corpus2 = _corpus([entry2])
        glossary = extractor.extract([corpus1, corpus2])

        # Deduplicated: same (source, target) → single term, 2 contexts
        assert glossary.term_count == 1
        assert len(glossary.terms[0].contexts) == 2

    def test_dedup_same_source_different_target(self, extractor: TermExtractor) -> None:
        """Same source, different target → separate terms."""
        hdr = _header(
            "Aid X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        entry1 = _bilingual(
            "key1",
            _entry("LocFriendlyName", "Aid Protocol"),
            _entry("LocFriendlyName", "援助协议"),
            hdr,
        )
        hdr2 = _header(
            "Aid2 X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        entry2 = _bilingual(
            "key2",
            _entry("LocFriendlyName", "Aid Protocol"),
            _entry("LocFriendlyName", "支援协议"),
            hdr2,
        )
        corpus = _corpus([entry1, entry2])
        glossary = extractor.extract([corpus])

        # Different targets → 2 separate terms
        aid_terms = [t for t in glossary.terms if t.source == "Aid Protocol"]
        assert len(aid_terms) == 2

    def test_dedup_merges_higher_priority_category(
        self, extractor: TermExtractor
    ) -> None:
        """When same (source, target) hit different rules, higher priority wins."""
        # Rule A match
        hdr_a = _header(
            "Rend X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        entry_a = _bilingual(
            "key1",
            _entry("LocFriendlyName", "Rend"),
            _entry("LocFriendlyName", "撕裂"),
            hdr_a,
        )

        # Rule C match (same source+target, different section)
        hdr_c = _header("SomeSection")
        entry_c = _bilingual(
            "key2",
            _entry("SomeKey", "Rend"),
            _entry("SomeKey", "撕裂"),
            hdr_c,
        )
        corpus = _corpus([entry_a, entry_c])
        glossary = extractor.extract([corpus])

        rend_terms = [t for t in glossary.terms if t.source == "Rend"]
        assert len(rend_terms) == 1
        assert rend_terms[0].category == "ability"  # Rule A (priority 0) wins over C

    def test_empty_corpora(self, extractor: TermExtractor) -> None:
        glossary = extractor.extract([])
        assert glossary.term_count == 0
        assert glossary.terms == []

    def test_sorting_by_category_and_source(self, extractor: TermExtractor) -> None:
        hdr_ability = _header(
            "B X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        hdr_weapon = _header(
            "A X2WeaponTemplate",
            class_name="X2WeaponTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        entries = [
            _bilingual(
                "k1",
                _entry("FriendlyName", "Rifle"),
                _entry("FriendlyName", "步枪"),
                hdr_weapon,
            ),
            _bilingual(
                "k2",
                _entry("LocFriendlyName", "Aid"),
                _entry("LocFriendlyName", "援助"),
                hdr_ability,
            ),
        ]
        corpus = _corpus(entries)
        glossary = extractor.extract([corpus])

        categories = [t.category for t in glossary.terms if not t.do_not_translate]
        assert categories == sorted(categories)


class TestPlaceholders:
    """Tests for Rule D placeholder collection."""

    def test_placeholders_collected(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        ph = PlaceholderSchema(
            pattern="<XGParam:IntValue0/>", type=PlaceholderType.XGPARAM, span=(0, 20)
        )
        src = _entry("Key", "Value <XGParam:IntValue0/>", placeholders=[ph])
        entry = _bilingual("key", src, None, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        ph_terms = [t for t in glossary.terms if t.do_not_translate]
        assert len(ph_terms) == 1
        assert ph_terms[0].source == "<XGParam:IntValue0/>"
        assert ph_terms[0].target == "<XGParam:IntValue0/>"
        assert ph_terms[0].category == "placeholder"
        assert ph_terms[0].same_as_source is True
        assert ph_terms[0].contexts == []

    def test_placeholders_deduped(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        ph = PlaceholderSchema(
            pattern="<Bullet/>", type=PlaceholderType.BULLET, span=(0, 8)
        )
        src1 = _entry("K1", "<Bullet/> First", placeholders=[ph])
        src2 = _entry("K2", "<Bullet/> Second", placeholders=[ph])
        entries = [
            _bilingual("k1", src1, None, hdr),
            _bilingual("k2", src2, None, hdr),
        ]
        corpus = _corpus(entries)
        glossary = extractor.extract([corpus])

        ph_terms = [t for t in glossary.terms if t.do_not_translate]
        assert len(ph_terms) == 1

    def test_multiple_placeholder_types(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        ph1 = PlaceholderSchema(
            pattern="<Bullet/>", type=PlaceholderType.BULLET, span=(0, 8)
        )
        ph2 = PlaceholderSchema(pattern="<br/>", type=PlaceholderType.BR, span=(10, 15))
        src = _entry("Key", "<Bullet/> x<br/>y", placeholders=[ph1, ph2])
        entry = _bilingual("key", src, None, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        ph_terms = [t for t in glossary.terms if t.do_not_translate]
        assert len(ph_terms) == 2
        patterns = {t.source for t in ph_terms}
        assert patterns == {"<Bullet/>", "<br/>"}

    def test_placeholders_sorted_after_regular(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        ph = PlaceholderSchema(
            pattern="<Bullet/>", type=PlaceholderType.BULLET, span=(0, 8)
        )
        # Entry with placeholder (contributes placeholder term only)
        src_ph = _entry("Key", "OK <Bullet/>", placeholders=[ph])
        entry_ph = _bilingual("key1", src_ph, None, hdr)
        # Entry without placeholder (contributes regular term)
        src_ok = _entry("Key2", "OK")
        entry_ok = _bilingual("key2", src_ok, _entry("Key2", "确定"), hdr)
        corpus = _corpus([entry_ph, entry_ok])
        glossary = extractor.extract([corpus])

        assert glossary.term_count >= 2
        # Last term should be placeholder
        assert glossary.terms[-1].do_not_translate is True


class TestEdgeCases:
    """Tests for boundary conditions."""

    def test_target_none_empty_target_text(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Aid X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocFriendlyName", "Aid Protocol")
        entry = _bilingual("key", src, None, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        term = next(t for t in glossary.terms if t.source == "Aid Protocol")
        assert term.target == ""
        assert term.same_as_source is False

    def test_same_as_source_marked(self, extractor: TermExtractor) -> None:
        hdr = _header("Section")
        src = _entry("Key", "HP")
        tgt = _entry("Key", "HP")
        entry = _bilingual("key", src, tgt, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        term = next(t for t in glossary.terms if t.source == "HP")
        assert term.same_as_source is True

    def test_append_struct_skipped(self, extractor: TermExtractor) -> None:
        from src.models.entry import StructFieldSchema

        hdr = _header("MissionObjectiveTexts")
        fields = [
            StructFieldSchema(
                key="MissionFamily", raw_value='"Recover"', value="Recover"
            ),
            StructFieldSchema(key="Description", raw_value='"text"', value="text"),
        ]
        src = _entry(
            "MissionDescriptions",
            "(MissionFamily=Recover, Description=text)",
            is_append=True,
            struct_fields=fields,
        )
        entry = _bilingual("key", src, None, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        # Append struct entry should be skipped entirely
        non_ph = [t for t in glossary.terms if not t.do_not_translate]
        assert len(non_ph) == 0

    def test_empty_value_skipped(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Rend X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocFriendlyName", "")
        entry = _bilingual("key", src, None, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        non_ph = [t for t in glossary.terms if not t.do_not_translate]
        assert len(non_ph) == 0

    def test_cosmetic_included_by_default(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Hair_A X2BodyPartTemplate",
            class_name="X2BodyPartTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("DisplayName", "Mohawk")
        entry = _bilingual("key", src, None, hdr)
        corpus = _corpus([entry])
        glossary = extractor.extract([corpus])

        assert any(t.category == "cosmetic" for t in glossary.terms)

    def test_context_tracks_source_path(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "Aid X2AbilityTemplate",
            class_name="X2AbilityTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("LocFriendlyName", "Aid")
        entry = _bilingual("key", src, _entry("LocFriendlyName", "援助"), hdr)
        corpus = BilingualCorpus(
            source_lang="en",
            target_lang="zh_Hans",
            source_path=Path("/data/XComGame.int"),
            target_path=Path("/data/XComGame.chn"),
            entries=[entry],
            namespace="test-extractor",
            mod_title="Test Extractor Fixture",
        )
        glossary = extractor.extract([corpus])

        term = glossary.terms[0]
        assert len(term.contexts) == 1
        assert term.contexts[0].source_path == Path("/data/XComGame.int")
        assert term.contexts[0].compound_key == "key"
        assert term.contexts[0].section_raw == "Aid X2AbilityTemplate"


class TestExtractorQuoteUnescape:
    """Parity with CorpusConverter: glossary terms must be unescaped the
    same way corpus units are before the Weblate boundary, or translators
    would see `\\"Betos\\"` in one view and `"Betos"` in another for the
    same source string."""

    def test_source_backslash_quote_unescaped(self, extractor: TermExtractor) -> None:
        hdr = _header(
            "UFOP_POI_BETOS X2EncyclopediaTemplate",
            class_name="X2EncyclopediaTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        # Parser-native form: value contains literal `\"` (the escape
        # for a dialogue quote inside the quoted string literal).
        src = _entry("ListTitle", '\\"Betos\\"')
        tgt = _entry("ListTitle", '\\"贝托斯\\"')
        corpus = _corpus([_bilingual("key", src, tgt, hdr)])

        glossary = extractor.extract([corpus])

        term = next(t for t in glossary.terms if "Betos" in t.source)
        assert term.source == '"Betos"'
        assert term.target == '"贝托斯"'
        # After unescape, identical source/target remain non-same
        assert term.same_as_source is False

    def test_same_as_source_detection_after_unescape(
        self, extractor: TermExtractor
    ) -> None:
        """same_as_source compares the UNESCAPED texts — so `\\"X\\"` in
        both sides collapses to `"X"` and is correctly flagged.
        """
        hdr = _header(
            "POI X2EncyclopediaTemplate",
            class_name="X2EncyclopediaTemplate",
            fmt=SectionHeaderFormat.OBJECT_CLASS,
        )
        src = _entry("ListTitle", '\\"Geist\\"')
        tgt = _entry("ListTitle", '\\"Geist\\"')
        corpus = _corpus([_bilingual("key", src, tgt, hdr)])

        glossary = extractor.extract([corpus])

        term = next(t for t in glossary.terms if "Geist" in t.source)
        assert term.source == '"Geist"'
        assert term.target == '"Geist"'
        assert term.same_as_source is True
