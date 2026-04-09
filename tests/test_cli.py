"""CLI integration tests using typer.testing.CliRunner."""

import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli.app import app
from tests.conftest import _write_loc_file

runner = CliRunner()


class TestParseCommand:
    def test_parse_json_stdout(self, sample_int: Path) -> None:
        result = runner.invoke(app, ["parse", str(sample_int)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "sections" in data
        assert data["lang"] == "en"

    def test_parse_csv_stdout(self, sample_int: Path) -> None:
        result = runner.invoke(app, ["parse", str(sample_int), "--format", "csv"])
        assert result.exit_code == 0
        lines = result.stdout.strip().splitlines()
        assert lines[0].startswith("section,key,")
        assert len(lines) > 1

    def test_parse_output_file(self, sample_int: Path, tmp_path: Path) -> None:
        out = tmp_path / "parsed.json"
        result = runner.invoke(app, ["parse", str(sample_int), "--output", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["lang"] == "en"

    def test_parse_file_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["parse", str(tmp_path / "nonexistent.int")])
        assert result.exit_code != 0


class TestAlignCommand:
    def test_align_json_stdout(
        self, align_source_int: Path, align_target_chn: Path
    ) -> None:
        result = runner.invoke(
            app, ["align", str(align_source_int), str(align_target_chn)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["source_lang"] == "en"
        assert data["target_lang"] == "zh_Hans"
        assert data["aligned_count"] > 0

    def test_align_csv_stdout(
        self, align_source_int: Path, align_target_chn: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "align",
                str(align_source_int),
                str(align_target_chn),
                "--format",
                "csv",
            ],
        )
        assert result.exit_code == 0
        lines = result.stdout.strip().splitlines()
        assert lines[0].startswith("compound_key,")

    def test_align_output_file(
        self, align_source_int: Path, align_target_chn: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "aligned.json"
        result = runner.invoke(
            app,
            [
                "align",
                str(align_source_int),
                str(align_target_chn),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert out.exists()


class TestAlignDirCommand:
    def test_align_dir_basic(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "INT"
        tgt_dir = tmp_path / "CHN"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        tgt_dir.mkdir()

        _write_loc_file(src_dir / "Test.int", '[Section]\nKey="Hello"')
        _write_loc_file(tgt_dir / "Test.chn", '[Section]\nKey="你好"')

        result = runner.invoke(
            app,
            [
                "align-dir",
                str(src_dir),
                str(tgt_dir),
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "Test.json").exists()

    def test_align_dir_with_target_lang(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "INT"
        tgt_dir = tmp_path / "CHN"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        tgt_dir.mkdir()

        _write_loc_file(src_dir / "Test.int", '[S]\nK="Hello"')

        result = runner.invoke(
            app,
            [
                "align-dir",
                str(src_dir),
                str(tgt_dir),
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "Test.json").exists()

    def test_align_dir_csv_format(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "INT"
        tgt_dir = tmp_path / "CHN"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        tgt_dir.mkdir()

        _write_loc_file(src_dir / "Test.int", '[S]\nK="Hello"')
        _write_loc_file(tgt_dir / "Test.chn", '[S]\nK="你好"')

        result = runner.invoke(
            app,
            [
                "align-dir",
                str(src_dir),
                str(tgt_dir),
                "--output-dir",
                str(out_dir),
                "--format",
                "csv",
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "Test.csv").exists()

    def test_align_dir_nonexistent_source(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["align-dir", str(tmp_path / "nope"), str(tmp_path)],
        )
        assert result.exit_code != 0


def _create_corpus_dir(base: Path) -> Path:
    """Create a temp directory with a corpus JSON file for extract tests."""
    import json as _json

    corpus_dir = base / "corpus"
    corpus_dir.mkdir(parents=True)

    corpus_data = {
        "source_lang": "en",
        "target_lang": "zh_Hans",
        "source_path": "/src/test.int",
        "target_path": "/tgt/test.chn",
        "entries": [
            {
                "compound_key": "Rend X2AbilityTemplate::LocFriendlyName",
                "source": {
                    "key": "LocFriendlyName",
                    "raw_value": '"Rend"',
                    "value": "Rend",
                    "is_array": False,
                    "array_index": None,
                    "is_append": False,
                    "struct_fields": None,
                    "placeholders": [],
                    "comments": [],
                    "line_number": 1,
                },
                "target": {
                    "key": "LocFriendlyName",
                    "raw_value": '"撕裂"',
                    "value": "撕裂",
                    "is_array": False,
                    "array_index": None,
                    "is_append": False,
                    "struct_fields": None,
                    "placeholders": [],
                    "comments": [],
                    "line_number": 1,
                },
                "section_header": {
                    "raw": "Rend X2AbilityTemplate",
                    "format": "object_class",
                    "name": "Rend X2AbilityTemplate",
                    "object_name": "Rend",
                    "class_name": "X2AbilityTemplate",
                    "package": None,
                },
            },
            {
                "compound_key": "UIUtilities_Text::m_strGenericOK",
                "source": {
                    "key": "m_strGenericOK",
                    "raw_value": '"OK"',
                    "value": "OK",
                    "is_array": False,
                    "array_index": None,
                    "is_append": False,
                    "struct_fields": None,
                    "placeholders": [
                        {
                            "pattern": "<Bullet/>",
                            "type": "bullet",
                            "span": [0, 8],
                        }
                    ],
                    "comments": [],
                    "line_number": 2,
                },
                "target": {
                    "key": "m_strGenericOK",
                    "raw_value": '"确定"',
                    "value": "确定",
                    "is_array": False,
                    "array_index": None,
                    "is_append": False,
                    "struct_fields": None,
                    "placeholders": [],
                    "comments": [],
                    "line_number": 2,
                },
                "section_header": {
                    "raw": "UIUtilities_Text",
                    "format": "class_only",
                    "name": "UIUtilities_Text",
                    "object_name": None,
                    "class_name": "UIUtilities_Text",
                    "package": None,
                },
            },
        ],
        "source_only": [],
        "target_only": [],
    }

    (corpus_dir / "test.json").write_text(
        _json.dumps(corpus_data, ensure_ascii=False), encoding="utf-8"
    )
    return corpus_dir


class TestExtractCommand:
    def test_extract_csv_stdout(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path)
        result = runner.invoke(app, ["extract", str(corpus_dir)])
        assert result.exit_code == 0
        assert "source,target" in result.stdout
        assert "Rend" in result.stdout

    def test_extract_json_output(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path)
        out = tmp_path / "glossary.json"
        result = runner.invoke(
            app,
            ["extract", str(corpus_dir), "--format", "json", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["term_count"] > 0

    def test_extract_csv_output(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path)
        out = tmp_path / "glossary.csv"
        result = runner.invoke(app, ["extract", str(corpus_dir), "--output", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        raw = out.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")

    def test_extract_multiple_dirs(self, tmp_path: Path) -> None:
        dir1 = _create_corpus_dir(tmp_path / "d1")
        dir2 = _create_corpus_dir(tmp_path / "d2")
        result = runner.invoke(app, ["extract", str(dir1), str(dir2)])
        assert result.exit_code == 0

    def test_extract_exclude_cosmetic(self, tmp_path: Path) -> None:
        corpus_dir = _create_corpus_dir(tmp_path)
        result = runner.invoke(
            app,
            ["extract", str(corpus_dir), "--format", "json", "--exclude-cosmetic"],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert all(t["category"] != "cosmetic" for t in data["terms"])

    def test_extract_nonexistent_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["extract", str(tmp_path / "nope")])
        assert result.exit_code != 0

    def test_extract_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["extract", str(empty)])
        assert result.exit_code != 0
