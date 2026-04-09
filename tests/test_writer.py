"""Unit tests for CorpusWriter."""

import csv
import io
import json
from pathlib import Path

import pytest

from src.export.writer import (
    CSV_COLUMNS,
    GLOSSARY_CSV_COLUMNS,
    CorpusWriter,
    GlossaryWriter,
)
from src.models._share import SectionHeaderFormat
from src.models.corpus import BilingualCorpus, BilingualEntry
from src.models.entry import EntrySchema
from src.models.glossary import Glossary, GlossaryTerm, TermContext
from src.models.section import SectionHeader


def _entry(key: str, value: str, line: int = 1, is_append: bool = False) -> EntrySchema:
    return EntrySchema(
        key=key,
        raw_value=f'"{value}"',
        value=value,
        is_append=is_append,
        line_number=line,
    )


def _header(raw: str) -> SectionHeader:
    return SectionHeader(
        raw=raw,
        format=SectionHeaderFormat.CLASS_ONLY,
        name=raw,
        class_name=raw,
    )


def _corpus(
    entries: list[BilingualEntry] | None = None,
    source_only: list[str] | None = None,
    target_only: list[str] | None = None,
) -> BilingualCorpus:
    return BilingualCorpus(
        source_lang="en",
        target_lang="zh_Hans",
        source_path=Path("/src/test.int"),
        target_path=Path("/tgt/test.chn"),
        entries=entries or [],
        source_only=source_only or [],
        target_only=target_only or [],
    )


@pytest.fixture
def writer() -> CorpusWriter:
    return CorpusWriter()


class TestCsvWriter:
    def test_column_order(self, writer: CorpusWriter) -> None:
        corpus = _corpus()
        result = writer.to_csv_string(corpus)
        header_line = result.splitlines()[0]
        assert header_line == ",".join(CSV_COLUMNS)

    def test_aligned_entry(self, writer: CorpusWriter) -> None:
        entry = BilingualEntry(
            compound_key="S::K",
            source=_entry("K", "hello", line=3),
            target=_entry("K", "你好", line=5),
            section_header=_header("S"),
        )
        corpus = _corpus(entries=[entry])
        result = writer.to_csv_string(corpus)

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["compound_key"] == "S::K"
        assert row["source_value"] == "hello"
        assert row["target_value"] == "你好"
        assert row["source_line"] == "3"
        assert row["target_line"] == "5"
        assert row["status"] == "aligned"

    def test_source_only_entry(self, writer: CorpusWriter) -> None:
        entry = BilingualEntry(
            compound_key="S::Missing",
            source=_entry("Missing", "no target"),
            target=None,
            section_header=_header("S"),
        )
        corpus = _corpus(entries=[entry], source_only=["S::Missing"])
        result = writer.to_csv_string(corpus)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["target_value"] == ""
        assert row["target_line"] == ""
        assert row["status"] == "source_only"

    def test_target_only_entry(self, writer: CorpusWriter) -> None:
        src_e = _entry("Extra", "placeholder_src", line=9)
        tgt_e = _entry("Extra", "额外", line=10)
        entry = BilingualEntry(
            compound_key="S::Extra",
            source=src_e,
            target=tgt_e,
            section_header=_header("S"),
        )
        corpus = _corpus(entries=[entry], target_only=["S::Extra"])
        result = writer.to_csv_string(corpus)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["source_value"] == "placeholder_src"
        assert row["target_value"] == "额外"
        assert row["status"] == "target_only"

    def test_empty_corpus(self, writer: CorpusWriter) -> None:
        corpus = _corpus()
        result = writer.to_csv_string(corpus)
        lines = result.strip().splitlines()
        assert len(lines) == 1  # header only

    def test_write_csv_bom(self, writer: CorpusWriter, tmp_path: Path) -> None:
        corpus = _corpus()
        out = tmp_path / "test.csv"
        writer.write_csv(corpus, out)

        raw = out.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM

    def test_write_csv_creates_parent_dirs(
        self, writer: CorpusWriter, tmp_path: Path
    ) -> None:
        corpus = _corpus()
        out = tmp_path / "sub" / "dir" / "test.csv"
        writer.write_csv(corpus, out)
        assert out.exists()

    def test_multiple_entries_order(self, writer: CorpusWriter) -> None:
        entries = [
            BilingualEntry(
                compound_key="S::A",
                source=_entry("A", "alpha"),
                target=_entry("A", "甲"),
                section_header=_header("S"),
            ),
            BilingualEntry(
                compound_key="S::B",
                source=_entry("B", "bravo"),
                target=None,
                section_header=_header("S"),
            ),
        ]
        corpus = _corpus(entries=entries, source_only=["S::B"])
        result = writer.to_csv_string(corpus)

        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["compound_key"] == "S::A"
        assert rows[0]["status"] == "aligned"
        assert rows[1]["compound_key"] == "S::B"
        assert rows[1]["status"] == "source_only"


class TestJsonWriter:
    def test_structure(self, writer: CorpusWriter) -> None:
        entry = BilingualEntry(
            compound_key="S::K",
            source=_entry("K", "hello"),
            target=_entry("K", "你好"),
            section_header=_header("S"),
        )
        corpus = _corpus(entries=[entry])
        result = writer.to_json_string(corpus)
        data = json.loads(result)

        assert data["source_lang"] == "en"
        assert data["target_lang"] == "zh_Hans"
        assert data["aligned_count"] == 1
        assert len(data["entries"]) == 1
        assert data["entries"][0]["compound_key"] == "S::K"
        assert data["entries"][0]["source"]["value"] == "hello"
        assert data["entries"][0]["target"]["value"] == "你好"

    def test_roundtrip(self, writer: CorpusWriter) -> None:
        entry = BilingualEntry(
            compound_key="S::K",
            source=_entry("K", "val"),
            target=None,
            section_header=_header("S"),
        )
        corpus = _corpus(entries=[entry], source_only=["S::K"])
        json_str = writer.to_json_string(corpus)
        data = json.loads(json_str)

        assert data["source_only"] == ["S::K"]
        assert data["target_only"] == []
        assert data["entries"][0]["target"] is None

    def test_chinese_not_escaped(self, writer: CorpusWriter) -> None:
        entry = BilingualEntry(
            compound_key="S::K",
            source=_entry("K", "hello"),
            target=_entry("K", "你好世界"),
            section_header=_header("S"),
        )
        corpus = _corpus(entries=[entry])
        result = writer.to_json_string(corpus)

        assert "你好世界" in result
        assert "\\u" not in result

    def test_write_json_file(self, writer: CorpusWriter, tmp_path: Path) -> None:
        corpus = _corpus()
        out = tmp_path / "test.json"
        writer.write_json(corpus, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["source_lang"] == "en"

    def test_empty_corpus(self, writer: CorpusWriter) -> None:
        corpus = _corpus()
        result = writer.to_json_string(corpus)
        data = json.loads(result)

        assert data["entries"] == []
        assert data["aligned_count"] == 0


def _term(
    source: str,
    target: str,
    category: str = "ability",
    do_not_translate: bool = False,
    same_as_source: bool = False,
    contexts: list[TermContext] | None = None,
) -> GlossaryTerm:
    return GlossaryTerm(
        source=source,
        target=target,
        category=category,
        do_not_translate=do_not_translate,
        same_as_source=same_as_source,
        contexts=contexts or [],
    )


def _glossary(terms: list[GlossaryTerm] | None = None) -> Glossary:
    return Glossary(
        source_lang="en",
        target_lang="zh_Hans",
        terms=terms or [],
    )


@pytest.fixture
def glossary_writer() -> GlossaryWriter:
    return GlossaryWriter()


class TestGlossaryCsvWriter:
    def test_column_order(self, glossary_writer: GlossaryWriter) -> None:
        glossary = _glossary()
        result = glossary_writer.to_csv_string(glossary)
        header_line = result.splitlines()[0]
        assert header_line == ",".join(GLOSSARY_CSV_COLUMNS)

    def test_term_with_context(self, glossary_writer: GlossaryWriter) -> None:
        ctx = TermContext(
            compound_key="S::K",
            section_raw="SectionRaw",
            key="KeyName",
            source_path=Path("/src/file.int"),
        )
        term = _term("Rend", "撕裂", category="ability", contexts=[ctx])
        glossary = _glossary([term])
        result = glossary_writer.to_csv_string(glossary)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["source"] == "Rend"
        assert row["target"] == "撕裂"
        assert row["category"] == "ability"
        assert row["context_section"] == "SectionRaw"
        assert row["context_key"] == "KeyName"
        assert row["context_source_file"] == "/src/file.int"
        assert row["context_count"] == "1"

    def test_do_not_translate_flag(self, glossary_writer: GlossaryWriter) -> None:
        term = _term(
            "<Bullet/>", "<Bullet/>", category="placeholder", do_not_translate=True
        )
        glossary = _glossary([term])
        result = glossary_writer.to_csv_string(glossary)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["do_not_translate"] == "true"

    def test_same_as_source_flag(self, glossary_writer: GlossaryWriter) -> None:
        term = _term("HP", "HP", same_as_source=True)
        glossary = _glossary([term])
        result = glossary_writer.to_csv_string(glossary)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["same_as_source"] == "true"

    def test_empty_flags_when_false(self, glossary_writer: GlossaryWriter) -> None:
        term = _term("Rend", "撕裂")
        glossary = _glossary([term])
        result = glossary_writer.to_csv_string(glossary)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["do_not_translate"] == ""
        assert row["same_as_source"] == ""

    def test_no_context_empty_fields(self, glossary_writer: GlossaryWriter) -> None:
        term = _term("<br/>", "<br/>", category="placeholder", do_not_translate=True)
        glossary = _glossary([term])
        result = glossary_writer.to_csv_string(glossary)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["context_section"] == ""
        assert row["context_key"] == ""
        assert row["context_source_file"] == ""
        assert row["context_count"] == "0"

    def test_merged_context_count(self, glossary_writer: GlossaryWriter) -> None:
        ctxs = [
            TermContext(
                compound_key=f"S::K{i}",
                section_raw="S",
                key=f"K{i}",
                source_path=Path("/f.int"),
            )
            for i in range(3)
        ]
        term = _term("Aid", "援助", contexts=ctxs)
        glossary = _glossary([term])
        result = glossary_writer.to_csv_string(glossary)

        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)
        assert row["context_count"] == "3"
        # First context used for section/key
        assert row["context_key"] == "K0"

    def test_write_csv_bom(
        self, glossary_writer: GlossaryWriter, tmp_path: Path
    ) -> None:
        glossary = _glossary()
        out = tmp_path / "glossary.csv"
        glossary_writer.write_csv(glossary, out)

        raw = out.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")

    def test_write_csv_creates_parent_dirs(
        self, glossary_writer: GlossaryWriter, tmp_path: Path
    ) -> None:
        glossary = _glossary()
        out = tmp_path / "sub" / "dir" / "glossary.csv"
        glossary_writer.write_csv(glossary, out)
        assert out.exists()

    def test_empty_glossary(self, glossary_writer: GlossaryWriter) -> None:
        glossary = _glossary()
        result = glossary_writer.to_csv_string(glossary)
        lines = result.strip().splitlines()
        assert len(lines) == 1  # header only


class TestGlossaryJsonWriter:
    def test_structure(self, glossary_writer: GlossaryWriter) -> None:
        term = _term("Rend", "撕裂", category="ability")
        glossary = _glossary([term])
        result = glossary_writer.to_json_string(glossary)
        data = json.loads(result)

        assert data["source_lang"] == "en"
        assert data["target_lang"] == "zh_Hans"
        assert data["term_count"] == 1
        assert len(data["terms"]) == 1
        assert data["terms"][0]["source"] == "Rend"
        assert data["terms"][0]["target"] == "撕裂"
        assert data["terms"][0]["category"] == "ability"

    def test_roundtrip(self, glossary_writer: GlossaryWriter) -> None:
        ctx = TermContext(
            compound_key="S::K",
            section_raw="S",
            key="K",
            source_path=Path("/f.int"),
        )
        term = _term("Aid", "援助", contexts=[ctx])
        glossary = _glossary([term])
        json_str = glossary_writer.to_json_string(glossary)
        data = json.loads(json_str)

        restored = Glossary.model_validate(data)
        assert restored.term_count == 1
        assert restored.terms[0].contexts[0].compound_key == "S::K"

    def test_chinese_not_escaped(self, glossary_writer: GlossaryWriter) -> None:
        term = _term("Rend", "撕裂")
        glossary = _glossary([term])
        result = glossary_writer.to_json_string(glossary)

        assert "撕裂" in result
        assert "\\u" not in result

    def test_write_json_file(
        self, glossary_writer: GlossaryWriter, tmp_path: Path
    ) -> None:
        glossary = _glossary()
        out = tmp_path / "glossary.json"
        glossary_writer.write_json(glossary, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["source_lang"] == "en"

    def test_empty_glossary(self, glossary_writer: GlossaryWriter) -> None:
        glossary = _glossary()
        result = glossary_writer.to_json_string(glossary)
        data = json.loads(result)

        assert data["terms"] == []
        assert data["term_count"] == 0
