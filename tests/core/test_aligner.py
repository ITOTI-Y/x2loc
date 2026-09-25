"""Unit tests for BilingualAligner."""

from pathlib import Path

import pytest

from src.core.aligner import BilingualAligner
from src.models._share import SectionHeaderFormat
from src.models.entry import EntrySchema, StructFieldSchema
from src.models.file import LocalizationFile
from src.models.section import SectionHeader, SectionSchema


def _make_entry(
    key: str,
    value: str = "",
    is_append: bool = False,
    line_number: int = 1,
    struct_fields: list[StructFieldSchema] | None = None,
) -> EntrySchema:
    return EntrySchema(
        key=key,
        raw_value=f'"{value}"',
        value=value,
        is_append=is_append,
        struct_fields=struct_fields,
        line_number=line_number,
    )


def _make_struct_entry(key: str, description: str) -> EntrySchema:
    """Helper: build a struct-append entry with a single Description field.

    Matches the real XCOM 2 shape `+Key=(Description="...")` and is what
    `iter_compound_keys_in_section` recognizes as a true append that carries
    an ordinal suffix on the compound key.
    """
    return EntrySchema(
        key=key,
        raw_value=f'(Description="{description}")',
        value=f'(Description="{description}")',
        is_append=True,
        struct_fields=[
            StructFieldSchema(
                key="Description",
                raw_value=f'"{description}"',
                value=description,
            )
        ],
        line_number=1,
    )


def _make_section(raw: str, entries: list[EntrySchema]) -> SectionSchema:
    return SectionSchema(
        header=SectionHeader(
            raw=raw,
            format=SectionHeaderFormat.CLASS_ONLY,
            name=raw,
            class_name=raw,
        ),
        entries=entries,
    )


def _make_file(
    sections: list[SectionSchema],
    lang: str = "en",
    path: str = "/tmp/test.int",
) -> LocalizationFile:
    return LocalizationFile(
        path=Path(path),
        lang=lang,
        sections=sections,
    )


class TestBuildIndex:
    def test_non_append_key(self, aligner: BilingualAligner) -> None:
        section = _make_section(
            "UIUtilities_Text",
            [_make_entry("m_strOK", "OK")],
        )
        file = _make_file([section])
        index = aligner._build_index(file)

        assert "UIUtilities_Text::m_strOK" in index
        entry, header = index["UIUtilities_Text::m_strOK"]
        assert entry.value == "OK"
        assert header.raw == "UIUtilities_Text"

    def test_struct_append_key_ordinal(self, aligner: BilingualAligner) -> None:
        """Struct append entries (`+Key=(...)`) retain the `#N` ordinal suffix
        because they represent genuine array-of-struct appends."""
        section = _make_section(
            "MissionSources X2MissionSourceTemplate",
            [
                _make_struct_entry("MissionDescriptions", "first"),
                _make_struct_entry("MissionDescriptions", "second"),
                _make_struct_entry("MissionDescriptions", "third"),
            ],
        )
        file = _make_file([section])
        index = aligner._build_index(file)

        assert len(index) == 3
        k0 = "MissionSources X2MissionSourceTemplate::MissionDescriptions#0"
        k1 = "MissionSources X2MissionSourceTemplate::MissionDescriptions#1"
        k2 = "MissionSources X2MissionSourceTemplate::MissionDescriptions#2"
        for key, expected in [(k0, "first"), (k1, "second"), (k2, "third")]:
            entry = index[key][0]
            assert entry.struct_fields is not None
            assert entry.struct_fields[0].value == expected

    def test_scalar_append_normalizes_to_non_append(
        self, aligner: BilingualAligner
    ) -> None:
        """Scalar `+Key="value"` entries normalize to the same key as
        non-append entries — the `+` prefix is effectively a no-op for
        scalar string properties in UE3 config files. See iter_compound_keys
        docstring for rationale.
        """
        section = _make_section(
            "Section",
            [_make_entry("m_strUrgent", "alert", is_append=True)],
        )
        file = _make_file([section])
        index = aligner._build_index(file)

        # Key has no `#N` suffix even though the entry is marked is_append
        assert "Section::m_strUrgent" in index
        assert "Section::m_strUrgent#0" not in index

    def test_scalar_append_aligns_with_non_append(
        self, aligner: BilingualAligner
    ) -> None:
        """Regression for LW_Overhaul: source uses `m_strUrgent=...` but
        target uses `+m_strUrgent=...`. Both must resolve to the same
        compound key so the aligner pairs them instead of dumping both
        into source_only / target_only."""
        src_section = _make_section(
            "UIMission",
            [_make_entry("m_strUrgent", "SECURITY BREACH")],
        )
        tgt_section = _make_section(
            "UIMission",
            [_make_entry("m_strUrgent", "安全漏洞", is_append=True)],
        )
        src_file = _make_file([src_section])
        tgt_file = _make_file([tgt_section], lang="zh_Hans", path="/tmp/tgt.chn")
        corpus = aligner.align(src_file, tgt_file)

        assert corpus.aligned_count == 1
        assert corpus.source_only == []
        assert corpus.target_only == []
        aligned = corpus.entries[0]
        assert aligned.source.value == "SECURITY BREACH"
        assert aligned.target is not None
        assert aligned.target.value == "安全漏洞"

    def test_mixed_struct_append_and_non_append(
        self, aligner: BilingualAligner
    ) -> None:
        section = _make_section(
            "Section",
            [
                _make_entry("NormalKey", "val"),
                _make_struct_entry("AppendKey", "a1"),
                _make_struct_entry("AppendKey", "a2"),
            ],
        )
        file = _make_file([section])
        index = aligner._build_index(file)

        assert "Section::NormalKey" in index
        assert "Section::AppendKey#0" in index
        assert "Section::AppendKey#1" in index

    def test_duplicate_non_append_indexed_as_array(
        self, aligner: BilingualAligner
    ) -> None:
        """Repeated non-append keys within one section are treated as an
        array: each occurrence gets an ordinal suffix instead of the old
        "last wins" behavior. This preserves data when a file genuinely
        has two `Key="..."` lines meant as separate list entries.
        """
        section = _make_section(
            "Section",
            [
                _make_entry("DupKey", "first"),
                _make_entry("DupKey", "second"),
            ],
        )
        file = _make_file([section])
        index = aligner._build_index(file)

        assert "Section::DupKey#0" in index
        assert "Section::DupKey#1" in index
        assert "Section::DupKey" not in index
        assert index["Section::DupKey#0"][0].value == "first"
        assert index["Section::DupKey#1"][0].value == "second"

    def test_multiple_sections(self, aligner: BilingualAligner) -> None:
        s1 = _make_section("SectionA", [_make_entry("Key1", "a")])
        s2 = _make_section("SectionB", [_make_entry("Key1", "b")])
        file = _make_file([s1, s2])
        index = aligner._build_index(file)

        assert index["SectionA::Key1"][0].value == "a"
        assert index["SectionB::Key1"][0].value == "b"

    def test_empty_file(self, aligner: BilingualAligner) -> None:
        file = _make_file([])
        index = aligner._build_index(file)
        assert index == {}

    def test_preserves_insertion_order(self, aligner: BilingualAligner) -> None:
        section = _make_section(
            "S",
            [
                _make_entry("Charlie", "c"),
                _make_entry("Alpha", "a"),
                _make_entry("Bravo", "b"),
            ],
        )
        file = _make_file([section])
        index = aligner._build_index(file)

        keys = list(index.keys())
        assert keys == ["S::Charlie", "S::Alpha", "S::Bravo"]


class TestAlign:
    def test_full_match(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [_make_section("S", [_make_entry("K", "hello")])],
            lang="en",
        )
        tgt = _make_file(
            [_make_section("S", [_make_entry("K", "你好")])],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.source_lang == "en"
        assert corpus.target_lang == "zh_Hans"
        assert corpus.aligned_count == 1
        assert corpus.source_only == []
        assert corpus.target_only == []
        assert len(corpus.entries) == 1
        assert corpus.entries[0].source.value == "hello"
        assert corpus.entries[0].target is not None
        assert corpus.entries[0].target.value == "你好"

    def test_source_only(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section(
                    "S",
                    [_make_entry("K1", "a"), _make_entry("K2", "b")],
                )
            ]
        )
        tgt = _make_file(
            [_make_section("S", [_make_entry("K1", "甲")])],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 1
        assert corpus.source_only == ["S::K2"]
        assert corpus.target_only == []

    def test_target_only(self, aligner: BilingualAligner) -> None:
        src = _make_file([_make_section("S", [_make_entry("K1", "a")])])
        tgt = _make_file(
            [
                _make_section(
                    "S",
                    [_make_entry("K1", "甲"), _make_entry("K2", "乙")],
                )
            ],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 1
        assert corpus.source_only == []
        assert corpus.target_only == ["S::K2"]
        # target_only entry appended to entries list
        assert len(corpus.entries) == 2
        tgt_only_entry = corpus.entries[1]
        assert tgt_only_entry.compound_key == "S::K2"
        # Aligner mirrors target into source for target_only entries
        assert tgt_only_entry.source is tgt_only_entry.target
        assert tgt_only_entry.target.value == "乙"

    def test_mixed(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section(
                    "S",
                    [
                        _make_entry("Shared", "shared"),
                        _make_entry("SrcOnly", "only_src"),
                    ],
                )
            ]
        )
        tgt = _make_file(
            [
                _make_section(
                    "S",
                    [
                        _make_entry("Shared", "共有"),
                        _make_entry("TgtOnly", "仅目标"),
                    ],
                )
            ],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 1
        assert corpus.source_only == ["S::SrcOnly"]
        assert corpus.target_only == ["S::TgtOnly"]
        assert len(corpus.entries) == 3

    def test_empty_source_and_target(self, aligner: BilingualAligner) -> None:
        src = _make_file([])
        tgt = _make_file([], lang="zh_Hans", path="/tmp/test.chn")
        corpus = aligner.align(src, tgt)

        assert corpus.entries == []
        assert corpus.source_only == []
        assert corpus.target_only == []
        assert corpus.aligned_count == 0

    def test_target_none_source_only_corpus(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section(
                    "S",
                    [_make_entry("K1", "a"), _make_entry("K2", "b")],
                )
            ]
        )
        corpus = aligner.align(src, target_lang="zh_Hans")

        assert corpus.target_lang == "zh_Hans"
        assert corpus.target_path is None
        assert corpus.aligned_count == 0
        assert len(corpus.source_only) == 2
        assert corpus.target_only == []

    def test_target_none_no_lang_raises(self, aligner: BilingualAligner) -> None:
        src = _make_file([])
        with pytest.raises(ValueError, match="target_lang"):
            aligner.align(src)

    def test_different_section_order(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section("A", [_make_entry("K", "a")]),
                _make_section("B", [_make_entry("K", "b")]),
            ]
        )
        tgt = _make_file(
            [
                _make_section("B", [_make_entry("K", "乙")]),
                _make_section("A", [_make_entry("K", "甲")]),
            ],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 2
        assert corpus.source_only == []
        assert corpus.target_only == []
        # Source order preserved
        assert corpus.entries[0].compound_key == "A::K"
        assert corpus.entries[0].target is not None
        assert corpus.entries[0].target.value == "甲"
        assert corpus.entries[1].compound_key == "B::K"
        assert corpus.entries[1].target is not None
        assert corpus.entries[1].target.value == "乙"

    def test_self_align(self, aligner: BilingualAligner) -> None:
        file = _make_file(
            [
                _make_section(
                    "S",
                    [_make_entry("K1", "a"), _make_entry("K2", "b")],
                )
            ]
        )
        corpus = aligner.align(file, file)

        assert corpus.aligned_count == 2
        assert corpus.source_only == []
        assert corpus.target_only == []

    def test_mod_info_flows_into_corpus(self, aligner: BilingualAligner) -> None:
        """align(mod_info=...) should populate namespace/steam_id/mod_title."""
        from src.models.mod import ModInfoSchema

        src = _make_file([_make_section("S", [_make_entry("K", "hello")])])
        tgt = _make_file(
            [_make_section("S", [_make_entry("K", "你好")])],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        mod = ModInfoSchema(
            namespace="1122837889-more-traits",
            steam_id="1122837889",
            mod_title="More Traits",
        )
        corpus = aligner.align(src, tgt, mod_info=mod)

        assert corpus.namespace == "1122837889-more-traits"
        assert corpus.steam_id == "1122837889"
        assert corpus.mod_title == "More Traits"

    def test_mod_info_default_is_base_game(self, aligner: BilingualAligner) -> None:
        """Without mod_info the corpus defaults to the base-game namespace."""
        from src.models.mod import BASE_GAME_NAMESPACE

        src = _make_file([_make_section("S", [_make_entry("K", "hello")])])
        tgt = _make_file(
            [_make_section("S", [_make_entry("K", "你好")])],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.namespace == BASE_GAME_NAMESPACE
        assert corpus.steam_id is None


class TestAppendAlignment:
    """Tests for true struct-append (`+Key=(...)`) alignment.

    Scalar append entries (`+Key="value"`) are covered separately in
    TestBuildIndex because they normalize to the same key shape as
    non-append entries.
    """

    def test_equal_struct_append_count(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section(
                    "M",
                    [
                        _make_struct_entry("Desc", "first"),
                        _make_struct_entry("Desc", "second"),
                    ],
                )
            ]
        )
        tgt = _make_file(
            [
                _make_section(
                    "M",
                    [
                        _make_struct_entry("Desc", "第一"),
                        _make_struct_entry("Desc", "第二"),
                    ],
                )
            ],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 2
        assert corpus.source_only == []
        assert corpus.target_only == []

    def test_source_more_struct_append(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section(
                    "M",
                    [
                        _make_struct_entry("D", "s0"),
                        _make_struct_entry("D", "s1"),
                        _make_struct_entry("D", "s2"),
                    ],
                )
            ]
        )
        tgt = _make_file(
            [
                _make_section(
                    "M",
                    [
                        _make_struct_entry("D", "t0"),
                        _make_struct_entry("D", "t1"),
                    ],
                )
            ],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 2
        assert corpus.source_only == ["M::D#2"]
        assert corpus.target_only == []

    def test_target_more_struct_append(self, aligner: BilingualAligner) -> None:
        src = _make_file(
            [
                _make_section("M", [_make_struct_entry("D", "s0")]),
            ]
        )
        tgt = _make_file(
            [
                _make_section(
                    "M",
                    [
                        _make_struct_entry("D", "t0"),
                        _make_struct_entry("D", "t1"),
                    ],
                )
            ],
            lang="zh_Hans",
            path="/tmp/test.chn",
        )
        corpus = aligner.align(src, tgt)

        assert corpus.aligned_count == 1
        assert corpus.source_only == []
        assert corpus.target_only == ["M::D#1"]
