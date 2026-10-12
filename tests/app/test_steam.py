from pathlib import Path

from src.app.steam import GAME_MODS, WORKSHOP, find_paths, library_paths

VDF = r"""
"libraryfolders"
{
    "0"
    {
        "path"		"C:\Program Files (x86)\Steam"
    }
    "1"
    {
        "path"		"F:\SteamLibrary"
    }
}
"""


def test_libraries_are_the_install_then_the_listed_folders() -> None:
    root = Path(r"C:\Program Files (x86)\Steam")

    assert library_paths(root, VDF) == [root, Path(r"F:\SteamLibrary")]


def test_paths_are_found_in_whichever_library_holds_them(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    (first / WORKSHOP).mkdir(parents=True)
    (second / GAME_MODS).mkdir(parents=True)

    found = find_paths([first, second])

    assert found.workshop_dir == first / WORKSHOP
    assert found.mods_dir == second / GAME_MODS
