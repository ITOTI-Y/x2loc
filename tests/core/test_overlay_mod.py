import zipfile
from pathlib import Path

from src.core.overlay_mod import install_overlay


def _zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in entries.items():
            archive.writestr(name, text)
    return path


def test_install_overlay_writes_manifest_and_replaces_previous(tmp_path: Path) -> None:
    mods_dir = tmp_path / "Mods"
    mods_dir.mkdir()
    first = _zip(tmp_path / "a.zip", {"Localization/Old.chn": "old"})
    second = _zip(tmp_path / "b.zip", {"Localization/Foo.chn": "new"})

    install_overlay(artifact=first, mods_dir=mods_dir, workshop_id="42", title="T")
    target = install_overlay(
        artifact=second, mods_dir=mods_dir, workshop_id="42", title="Test Mod"
    )

    assert target == mods_dir / "x2loc_zh_42"
    assert (target / "Localization" / "Foo.chn").read_text() == "new"
    assert not (target / "Localization" / "Old.chn").exists()
    manifest = (target / "x2loc_zh_42.XComMod").read_text(encoding="utf-8")
    assert "Title=Test Mod [zh_Hans]" in manifest
    assert sorted(p.name for p in mods_dir.iterdir()) == ["x2loc_zh_42"]
