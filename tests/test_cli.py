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
