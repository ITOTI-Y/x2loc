"""Find XCOM 2's Workshop content and the game's Mods folder via Steam."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.models.workshop import XCOM2_APP_ID

LIBRARY_PATH: Final = re.compile(r'^\s*"path"\s+"((?:[^"\\]|\\.)*)"', re.MULTILINE)
GAME_MODS: Final = Path("steamapps/common/XCOM 2/XCom2-WarOfTheChosen/XComGame/Mods")
WORKSHOP: Final = Path(f"steamapps/workshop/content/{XCOM2_APP_ID}")


@dataclass(frozen=True)
class DetectedPaths:
    workshop_dir: Path | None
    mods_dir: Path | None


def library_paths(steam_root: Path, vdf_text: str) -> list[Path]:
    """Every Steam library: the install itself, then `libraryfolders.vdf`'s."""
    paths = [steam_root]
    for raw in LIBRARY_PATH.findall(vdf_text):
        path = Path(raw.replace("\\\\", "\\"))
        if path not in paths:
            paths.append(path)
    return paths


def find_paths(libraries: list[Path]) -> DetectedPaths:
    def first(relative: Path) -> Path | None:
        return next(
            (lib / relative for lib in libraries if (lib / relative).is_dir()), None
        )

    return DetectedPaths(workshop_dir=first(WORKSHOP), mods_dir=first(GAME_MODS))


def steam_root() -> Path | None:
    """Steam's install folder from the registry; None off Windows or uninstalled."""
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _kind = winreg.QueryValueEx(key, "SteamPath")
    except OSError:
        return None
    return Path(value)


def detect_paths() -> DetectedPaths:
    root = steam_root()
    if root is None:
        return DetectedPaths(workshop_dir=None, mods_dir=None)
    vdf = root / "steamapps" / "libraryfolders.vdf"
    text = vdf.read_text(encoding="utf-8", errors="replace") if vdf.is_file() else ""
    return find_paths(library_paths(root, text))
