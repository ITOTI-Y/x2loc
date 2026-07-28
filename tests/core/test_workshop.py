from pathlib import Path

import pytest

from src.core.workshop import (
    WorkshopInputError,
    component_slug,
    discover_localization_assets,
    parse_workshop_url,
    scan_mod_tree,
)
from src.models.mod import ModInfoSchema
from src.models.workshop import WorkshopItemSchema, WorkshopLimitsSchema
from tests.conftest import _write_loc_file

VALID_URL = "https://steamcommunity.com/sharedfiles/filedetails/?id=1234567890"


def limits(**overrides: int) -> WorkshopLimitsSchema:
    values: dict[str, float] = {
        "download_timeout_seconds": 60.0,
        "terminate_grace_seconds": 5.0,
        "max_total_bytes": 10_000_000,
        "max_file_count": 100,
        "max_loc_file_bytes": 1_000_000,
    }
    values.update(overrides)
    return WorkshopLimitsSchema.model_validate(values)


def make_item(mod_root: Path, files: list[Path]) -> WorkshopItemSchema:
    return WorkshopItemSchema(
        workshop_id="1234567890",
        mod_root=mod_root,
        mod_info=ModInfoSchema(
            mod_title="Test Mod", namespace="1234567890-test-mod", steam_id="1234567890"
        ),
        files=files,
    )


def test_parse_workshop_url_returns_id() -> None:
    assert parse_workshop_url(VALID_URL) == "1234567890"


@pytest.mark.parametrize(
    "url",
    [
        "http://steamcommunity.com/sharedfiles/filedetails/?id=1",
        "https://evil.test/sharedfiles/filedetails/?id=1",
        "https://steamcommunity.com/sharedfiles/filedetails/?id=0",
        "https://steamcommunity.com/sharedfiles/filedetails/?id=abc",
        "https://steamcommunity.com/sharedfiles/filedetails/?id=1&id=2",
        "https://steamcommunity.com/workshop/browse/?id=1",
        "https://user:pw@steamcommunity.com/sharedfiles/filedetails/?id=1",
    ],
)
def test_parse_workshop_url_rejects_bad_input(url: str) -> None:
    with pytest.raises(WorkshopInputError):
        parse_workshop_url(url)


def test_component_slug_is_stable_and_path_scoped() -> None:
    from pathlib import PurePosixPath

    first = component_slug("42", PurePosixPath("Localization/Foo.int"))
    second = component_slug("42", PurePosixPath("Localization/Foo.int"))
    other = component_slug("42", PurePosixPath("Other/Foo.int"))
    assert first == second
    assert first != other
    assert first.startswith("mod-42-")


def test_scan_mod_tree_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    (root / "link.int").symlink_to(root / "Localization" / "Foo.int")
    with pytest.raises(WorkshopInputError, match="symbolic link"):
        scan_mod_tree(root, limits())


def test_scan_mod_tree_enforces_loc_file_size(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    with pytest.raises(WorkshopInputError, match="size limit"):
        scan_mod_tree(root, limits(max_loc_file_bytes=1))


def test_scan_mod_tree_enforces_file_count(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    for index in range(3):
        _write_loc_file(root / "Localization" / f"F{index}.int", '[S]\nK="V"')
    with pytest.raises(WorkshopInputError, match="file count"):
        scan_mod_tree(root, limits(max_file_count=2))


def test_discover_requires_localization_directory(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_loc_file(root / "Config" / "Foo.int", '[S]\nK="V"')
    files = scan_mod_tree(root, limits())
    with pytest.raises(WorkshopInputError, match=r"no \.int file"):
        discover_localization_assets(make_item(root.resolve(), files))


def test_discover_marks_existing_target(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    _write_loc_file(root / "Localization" / "Foo.chn", '[S]\nK="值"')
    files = scan_mod_tree(root, limits())
    assets = discover_localization_assets(make_item(root.resolve(), files))
    assert len(assets) == 1
    assert assets[0].existing_target_path is not None
    assert assets[0].relative_target_path.as_posix() == "Localization/Foo.chn"


def test_discover_rejects_windows_case_collision(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_loc_file(root / "Localization" / "Foo.int", '[S]\nK="V"')
    _write_loc_file(root / "Localization" / "FOO.int", '[S]\nK="V"')
    files = scan_mod_tree(root, limits())
    with pytest.raises(WorkshopInputError, match="collision"):
        discover_localization_assets(make_item(root.resolve(), files))
