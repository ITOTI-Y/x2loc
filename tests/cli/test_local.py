from pathlib import Path

import pytest
import typer

from src.cli.local import _load_items
from src.models.workshop import WorkshopLimitsSchema
from tests.conftest import _write_loc_file

LIMITS = WorkshopLimitsSchema(
    download_timeout_seconds=60.0,
    terminate_grace_seconds=5.0,
    max_total_bytes=10_000_000,
    max_file_count=100,
    max_loc_file_bytes=1_000_000,
)


def _write_mod(root: Path, *, localized: bool) -> None:
    root.mkdir(parents=True)
    (root / "TestMod.XComMod").write_text("[mod]\nTitle=Test Mod\n", encoding="utf-8")
    if localized:
        _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')


def test_load_items_skips_mod_without_localization(tmp_path: Path) -> None:
    _write_mod(tmp_path / "111", localized=True)
    _write_mod(tmp_path / "222", localized=False)
    (tmp_path / "not-a-mod").mkdir()

    items, works, skipped = _load_items(tmp_path, None, LIMITS)

    assert list(items) == ["111"]
    assert list(works) == ["111"] and works["111"]
    assert items["111"].mod_info.mod_title == "Test Mod"
    assert [entry["workshop_id"] for entry in skipped] == ["222"]


def test_load_items_rejects_unknown_id(tmp_path: Path) -> None:
    _write_mod(tmp_path / "111", localized=True)
    with pytest.raises(typer.BadParameter, match="999"):
        _load_items(tmp_path, ["111", "999"], LIMITS)
