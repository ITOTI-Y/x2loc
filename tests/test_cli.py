"""CLI integration tests using typer.testing.CliRunner."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
                "--base-game",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "_base" / "Test.json").exists()

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
                "--base-game",
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "_base" / "Test.json").exists()

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
                "--base-game",
                "--output-dir",
                str(out_dir),
                "--format",
                "csv",
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "_base" / "Test.csv").exists()

    def test_align_dir_nonexistent_source(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["align-dir", str(tmp_path / "nope"), str(tmp_path), "--base-game"],
        )
        assert result.exit_code != 0

    def test_align_dir_requires_sandbox_or_base_game(self, tmp_path: Path) -> None:
        """Without --sandbox-root and without --base-game, exit non-zero."""
        src_dir = tmp_path / "INT"
        tgt_dir = tmp_path / "CHN"
        src_dir.mkdir()
        tgt_dir.mkdir()
        _write_loc_file(src_dir / "Test.int", '[S]\nK="Hello"')

        result = runner.invoke(
            app,
            ["align-dir", str(src_dir), str(tgt_dir)],
        )
        assert result.exit_code != 0

    def test_align_dir_with_sandbox_and_mod_resolver(self, tmp_path: Path) -> None:
        """End-to-end: align-dir resolves the mod namespace from .XComMod
        and writes the corpus under output/{namespace}/."""
        mod_root = tmp_path / "1122837889"
        src_dir = mod_root / "Localization"
        tgt_dir = tmp_path / "translations"
        out_dir = tmp_path / "out"
        src_dir.mkdir(parents=True)
        tgt_dir.mkdir()

        (mod_root / "MoreTraits.XComMod").write_text(
            "[mod]\npublishedFileId=1122837889\nTitle=More Traits\n",
            encoding="utf-8",
        )
        _write_loc_file(src_dir / "Test.int", '[S]\nK="Hello"')
        _write_loc_file(tgt_dir / "Test.chn", '[S]\nK="你好"')

        result = runner.invoke(
            app,
            [
                "align-dir",
                str(src_dir),
                str(tgt_dir),
                "--sandbox-root",
                str(tmp_path),
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        expected = out_dir / "1122837889-more-traits" / "Test.json"
        assert expected.exists(), f"{expected} not found"


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
        "namespace": "test-extract-fixture",
        "mod_title": "Test Extract Fixture",
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


def _write_upload_corpus(corpus_dir: Path, stem: str) -> Path:
    """Write a minimal corpus JSON file for upload-command tests."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_lang": "en",
        "target_lang": "zh_Hans",
        "source_path": str(corpus_dir / f"{stem}.int"),
        "target_path": str(corpus_dir / f"{stem}.chn"),
        "namespace": "base-xcom2-wotc",
        "mod_title": "XCOM 2 War of the Chosen",
        "entries": [
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
                    "placeholders": [],
                    "comments": [],
                    "line_number": 1,
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
                    "line_number": 1,
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
    out = corpus_dir / f"{stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


class TestUploadCommand:
    @patch("src.cli.app.WeblateClient")
    def test_upload_creates_component_when_absent(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = {"slug": "test"}
        instance.get_component.return_value = None
        instance.create_component.return_value = {"slug": "XComGame"}

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
            ],
        )

        assert result.exit_code == 0, result.stdout
        instance.create_component.assert_called_once()
        instance.create_translation.assert_called()
        # Component was new, so translated CSV should be uploaded too
        instance.upload_file.assert_called()

    @patch("src.cli.app.WeblateClient")
    def test_upload_updates_existing_component(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """Re-upload to an existing component with default (replace) method."""
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = {"slug": "test"}
        instance.get_component.return_value = {
            "slug": "base-xcom2-wotc-XComGame",
            "stats": {"total": 1},
        }
        # Existing unit already covers the only context in the fixture,
        # so no create_unit calls should fire.
        instance.list_units.return_value = iter(
            [{"context": "UIUtilities_Text::m_strGenericOK"}]
        )

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
                "--yes",  # skip replace-confirmation prompt
            ],
        )

        assert result.exit_code == 0, result.stdout
        instance.create_component.assert_not_called()
        instance.create_unit.assert_not_called()
        # Existing component → upload_file called at least once for target
        assert instance.upload_file.call_count >= 1

    @patch("src.cli.app.WeblateClient")
    def test_upload_adds_new_units_to_existing_component(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """Mode 2: new contexts in the local corpus that don't exist yet
        in Weblate must be created via `create_unit`, not via a destructive
        replace upload."""
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = {"slug": "test"}
        instance.get_component.return_value = {
            "slug": "base-xcom2-wotc-XComGame",
            "stats": {"total": 0},
        }
        # Weblate reports zero existing units — the one unit in our fixture
        # must therefore be added via create_unit.
        instance.list_units.return_value = iter([])

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.stdout
        instance.create_component.assert_not_called()
        instance.create_unit.assert_called_once()
        # Inspect the create_unit call shape — for non-glossary bilingual
        # CSV components Weblate expects monolingual `key` + `value` shape
        # (value as a list), not glossary-style source/target.
        call = instance.create_unit.call_args
        assert call.args[0] == "base-xcom2-wotc-XComGame"
        assert call.args[1] == "en"  # source_lang from fixture
        body = call.args[2]
        assert body["key"] == "UIUtilities_Text::m_strGenericOK"
        assert body["value"] == ["OK"]

    @patch("src.cli.app.WeblateClient")
    def test_upload_glossary_creates_new_component_mode1(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """Glossary slug should be `glossary-{namespace}` and Mode 1 creates
        the component via `create_component` with the source CSV."""
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")
        glossary_csv = tmp_path / "glossary.csv"
        glossary_csv.write_text(
            "source,target,category,do_not_translate,same_as_source\n"
            "Chain Lightning,闪电链,ability,false,false\n",
            encoding="utf-8",
        )

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = {"slug": "test"}
        # First get_component call (corpus) → None, Mode 1 create. Second
        # (glossary) → None too. Third call is in _mark_glossary_flags
        # which fetches source_language metadata.
        instance.get_component.side_effect = [
            None,  # corpus component lookup
            None,  # glossary component lookup
            {"source_language": {"code": "en"}},  # metadata after create
        ]
        instance.list_units.return_value = iter([])

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--glossary",
                str(glossary_csv),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.stdout
        # create_component called twice: once for corpus, once for glossary
        assert instance.create_component.call_count == 2
        glossary_call = next(
            c
            for c in instance.create_component.call_args_list
            if c.kwargs.get("is_glossary")
        )
        assert glossary_call.kwargs["slug"] == "glossary-base-xcom2-wotc"

    @patch("src.cli.app.WeblateClient")
    def test_upload_glossary_adds_units_to_existing_mode2(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """Mode 2 for glossary: existing component should receive only
        the new term as a `create_unit` POST, not a full-file replace."""
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")
        glossary_csv = tmp_path / "glossary.csv"
        glossary_csv.write_text(
            "source,target,category,do_not_translate,same_as_source\n"
            "OldTerm,旧术语,ability,false,false\n"
            "NewTerm,新术语,ability,false,false\n",
            encoding="utf-8",
        )

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = {"slug": "test"}
        instance.get_component.side_effect = [
            None,  # corpus component (Mode 1, irrelevant here)
            {
                "slug": "glossary-base-xcom2-wotc",
                "file_format": "csv",
                "source_language": {"code": "en"},
            },
            # _mark_glossary_flags re-fetches metadata
            {"source_language": {"code": "en"}},
        ]
        # Corpus goes through Mode 1 (no list_units). Glossary Mode 2 and
        # _mark_glossary_flags each call list_units once per language pass.
        instance.list_units.side_effect = [
            iter([{"context": "OldTerm::ability"}]),  # glossary Mode 2 diff
            iter([]),  # _mark_glossary_flags source-lang pass
            iter([]),  # _mark_glossary_flags target-lang pass
        ]

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--glossary",
                str(glossary_csv),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.stdout
        # Glossary Mode 2: create_component must NOT be called for glossary.
        glossary_creates = [
            c
            for c in instance.create_component.call_args_list
            if c.kwargs.get("is_glossary")
        ]
        assert glossary_creates == []
        # Exactly one new glossary term was pushed via create_unit.
        glossary_unit_calls = [
            c
            for c in instance.create_unit.call_args_list
            if c.args[0] == "glossary-base-xcom2-wotc"
        ]
        assert len(glossary_unit_calls) == 1
        body = glossary_unit_calls[0].args[2]
        # Glossary on hosted.weblate.org (as of 2026-04) uses the
        # monolingual-template body shape, same as non-glossary bilingual.
        assert body["key"] == "NewTerm::ability"
        assert body["value"] == ["NewTerm"]

    @patch("src.cli.app.WeblateClient")
    def test_upload_creates_project_when_missing(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = None  # project missing
        instance.get_component.return_value = None
        instance.create_component.return_value = {"slug": "XComGame"}

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
            ],
        )

        assert result.exit_code == 0, result.stdout
        instance.create_project.assert_called_once()

    @patch("src.cli.app.WeblateClient")
    def test_upload_empty_corpus_dir_errors(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        instance = mock_weblate.return_value.__enter__.return_value
        instance.get_project.return_value = {"slug": "test"}

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--target-lang",
                "zh_Hans",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
            ],
        )
        assert result.exit_code != 0

    def test_upload_missing_credentials_errors(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")

        result = runner.invoke(
            app,
            [
                "upload",
                str(corpus_dir),
                "--target-lang",
                "zh_Hans",
                # no --url/--token/--project/--config
            ],
        )
        assert result.exit_code != 0

    def test_upload_config_file(self, tmp_path: Path) -> None:
        """Loading Weblate config from TOML file."""
        corpus_dir = tmp_path / "corpus"
        _write_upload_corpus(corpus_dir, "XComGame")

        config_path = tmp_path / "weblate.toml"
        config_path.write_text(
            'url = "https://weblate.example.com/api/"\n'
            'token = "wlp_from_file"\n'
            'project_slug = "from-file"\n',
            encoding="utf-8",
        )

        with patch("src.cli.app.WeblateClient") as mock_weblate:
            instance = mock_weblate.return_value.__enter__.return_value
            instance.get_project.return_value = {"slug": "from-file"}
            instance.get_component.return_value = None
            instance.create_component.return_value = {"slug": "XComGame"}

            result = runner.invoke(
                app,
                [
                    "upload",
                    str(corpus_dir),
                    "--target-lang",
                    "zh_Hans",
                    "--config",
                    str(config_path),
                ],
            )
            assert result.exit_code == 0, result.stdout
            # Verify the config file values flowed into the constructed schema
            cfg = mock_weblate.call_args.args[0]
            assert cfg.token == "wlp_from_file"
            assert cfg.project_slug == "from-file"


class TestDownloadCommand:
    """Namespace-filtered download. Acceptance criteria for this command
    are validated against real hosted.weblate.org (see docs/_dev/WEBLATE.md);
    these mock tests pin the namespace-prefix filtering logic and directory
    layout only.
    """

    @patch("src.cli.app.WeblateClient")
    def test_download_filters_by_namespace_prefix(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """Only components whose slug starts with `{namespace}-` are
        downloaded, and saved under `{output_dir}/{namespace}/{stem}.csv`.
        Glossary components and components from other namespaces are
        skipped even when they're in the project.
        """
        ns = "1122837889-more-traits"
        instance = mock_weblate.return_value.__enter__.return_value
        instance.list_components.return_value = [
            {"slug": f"{ns}-XComGame", "is_glossary": False},
            {"slug": f"{ns}-UIScreens", "is_glossary": False},
            {"slug": f"glossary-{ns}", "is_glossary": True},  # glossary skipped
            {
                "slug": "other-mod-XComGame",  # different namespace
                "is_glossary": False,
            },
        ]
        instance.download_file.return_value = b"context,source,target\n"

        out_dir = tmp_path / "translations"
        result = runner.invoke(
            app,
            [
                "download",
                "--namespace",
                ns,
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (out_dir / ns / "XComGame.csv").exists()
        assert (out_dir / ns / "UIScreens.csv").exists()
        # Namespace prefix was stripped from the filenames.
        assert not (out_dir / ns / f"{ns}-XComGame.csv").exists()
        assert not (out_dir / ns / f"glossary-{ns}.csv").exists()
        assert not (out_dir / ns / "other-mod-XComGame.csv").exists()

    @patch("src.cli.app.WeblateClient")
    def test_download_component_filter_stem_match(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """`--component` narrows to a single stem AFTER the namespace prefix
        is stripped (so callers pass `XComGame`, not the full slug)."""
        ns = "base-xcom2-wotc"
        instance = mock_weblate.return_value.__enter__.return_value
        instance.list_components.return_value = [
            {"slug": f"{ns}-XComGame", "is_glossary": False},
            {"slug": f"{ns}-UIScreens", "is_glossary": False},
        ]
        instance.download_file.return_value = b"context,source,target\n"

        out_dir = tmp_path / "translations"
        result = runner.invoke(
            app,
            [
                "download",
                "--namespace",
                ns,
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
                "--component",
                "XComGame",
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (out_dir / ns / "XComGame.csv").exists()
        assert not (out_dir / ns / "UIScreens.csv").exists()

    @patch("src.cli.app.WeblateClient")
    def test_download_empty_namespace_warns(
        self, mock_weblate: MagicMock, tmp_path: Path
    ) -> None:
        """Non-existent namespace exits 0 with a warning (not an error —
        a future job may upload into this namespace later)."""
        instance = mock_weblate.return_value.__enter__.return_value
        instance.list_components.return_value = [
            {"slug": "other-mod-XComGame", "is_glossary": False},
        ]

        out_dir = tmp_path / "translations"
        result = runner.invoke(
            app,
            [
                "download",
                "--namespace",
                "does-not-exist",
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
                "--url",
                "https://weblate.example.com/api/",
                "--token",
                "wlp_test",
                "--project",
                "test",
            ],
        )
        assert result.exit_code == 0
        instance.download_file.assert_not_called()


class TestWritebackCommand:
    def test_writeback_produces_chn_file(
        self, tmp_path: Path, struct_append_int: Path
    ) -> None:
        """End-to-end: source .int + translated CSV → .chn file."""
        src_dir = tmp_path / "INT"
        src_dir.mkdir()
        dest = src_dir / struct_append_int.name
        dest.write_bytes(struct_append_int.read_bytes())

        translations_dir = tmp_path / "csv"
        translations_dir.mkdir()
        csv_path = translations_dir / f"{dest.stem}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["context", "source", "target", "developer_comments"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "context": "Rend X2AbilityTemplate::LocFriendlyName",
                    "source": "Rend",
                    "target": "撕裂",
                    "developer_comments": "",
                }
            )

        out_dir = tmp_path / "CHN"
        result = runner.invoke(
            app,
            [
                "writeback",
                str(src_dir),
                str(translations_dir),
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        out_file = out_dir / f"{dest.stem}.chn"
        assert out_file.exists()
        raw = out_file.read_bytes()
        assert raw.startswith(b"\xff\xfe")
        # CRLF encoded in UTF-16-LE → \r\x00\n\x00
        assert b"\r\x00\n\x00" in raw

    def test_writeback_missing_csv_is_skipped(
        self, tmp_path: Path, struct_append_int: Path
    ) -> None:
        src_dir = tmp_path / "INT"
        src_dir.mkdir()
        (src_dir / struct_append_int.name).write_bytes(struct_append_int.read_bytes())

        empty_csv_dir = tmp_path / "empty"
        empty_csv_dir.mkdir()

        out_dir = tmp_path / "CHN"
        result = runner.invoke(
            app,
            [
                "writeback",
                str(src_dir),
                str(empty_csv_dir),
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
            ],
        )
        # Command should succeed but produce no output file
        assert result.exit_code == 0, result.stdout
        assert not list(out_dir.glob("*.chn"))

    def test_writeback_partial_translation_fallback(
        self, tmp_path: Path, struct_append_int: Path
    ) -> None:
        """Untranslated entries keep source values end-to-end."""
        src_dir = tmp_path / "INT"
        src_dir.mkdir()
        dest = src_dir / struct_append_int.name
        dest.write_bytes(struct_append_int.read_bytes())

        translations_dir = tmp_path / "csv"
        translations_dir.mkdir()
        csv_path = translations_dir / f"{dest.stem}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["context", "source", "target", "developer_comments"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "context": (
                        "MissionObjectiveTexts::MissionDescriptions#0::Description"
                    ),
                    "source": "We have discovered a supply cache.",
                    "target": "我们发现了一个补给缓存。",
                    "developer_comments": "",
                }
            )

        out_dir = tmp_path / "CHN"
        result = runner.invoke(
            app,
            [
                "writeback",
                str(src_dir),
                str(translations_dir),
                "--target-lang",
                "zh_Hans",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout

        # Re-parse the output and verify fallback entries retained source
        from src.core.parser import LocFileParser

        out_file = out_dir / f"{dest.stem}.chn"
        reparsed = LocFileParser().parse(out_file)
        rend_section = next(s for s in reparsed.sections if s.header.name == "Rend")
        loc_friendly = next(
            e for e in rend_section.entries if e.key == "LocFriendlyName"
        )
        assert loc_friendly.value == "Rend"  # untranslated → source preserved

    def test_writeback_missing_source_dir_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "writeback",
                str(tmp_path / "nope"),
                str(tmp_path),
                "--target-lang",
                "zh_Hans",
            ],
        )
        assert result.exit_code != 0

    def test_writeback_unknown_target_lang_errors(
        self, tmp_path: Path, struct_append_int: Path
    ) -> None:
        src_dir = tmp_path / "INT"
        src_dir.mkdir()
        (src_dir / struct_append_int.name).write_bytes(struct_append_int.read_bytes())
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "writeback",
                str(src_dir),
                str(csv_dir),
                "--target-lang",
                "klingon",
            ],
        )
        assert result.exit_code != 0
