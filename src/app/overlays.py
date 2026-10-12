"""The mods x2loc installs into the game's Mods folder."""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.core.overlay_mod import overlay_mod_name

OVERLAY_DIR: Final = re.compile(r"^x2loc_(\d+)_zh$")
TITLE_LINE: Final = re.compile(r"^Title=(.*)$", re.MULTILINE)
WEAPON_NAMES_MOD: Final = "x2locWeaponCatNames"
# What the game loads from a script mod; its sources stay in the repo.
WEAPON_NAMES_PARTS: Final = ("Config", "Script", f"{WEAPON_NAMES_MOD}.XComMod")


@dataclass(frozen=True)
class InstalledOverlay:
    workshop_id: str
    title: str


def list_overlays(mods_dir: Path) -> list[InstalledOverlay]:
    overlays: list[InstalledOverlay] = []
    for folder in sorted(mods_dir.iterdir()):
        match = OVERLAY_DIR.match(folder.name)
        if match is None or not folder.is_dir():
            continue
        manifest = folder / f"{folder.name}.XComMod"
        text = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
        title = TITLE_LINE.search(text)
        overlays.append(
            InstalledOverlay(match[1], title[1].strip() if title else folder.name)
        )
    return overlays


def uninstall_overlay(mods_dir: Path, workshop_id: str) -> None:
    """Remove one overlay; raises FileNotFoundError when none is installed."""
    folder = mods_dir / overlay_mod_name(workshop_id)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    shutil.rmtree(folder)


def weapon_names_installed(mods_dir: Path) -> bool:
    return (mods_dir / WEAPON_NAMES_MOD / f"{WEAPON_NAMES_MOD}.XComMod").is_file()


def install_weapon_names(bundle: Path, mods_dir: Path) -> None:
    """Copy the prebuilt weapon category mod from `bundle` into the game.

    Assembled beside its final place and swapped in whole, like overlays.
    """
    target = mods_dir / WEAPON_NAMES_MOD
    staging = mods_dir / f".{WEAPON_NAMES_MOD}.part"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for part in WEAPON_NAMES_PARTS:
        source = bundle / part
        if source.is_dir():
            shutil.copytree(source, staging / part)
        else:
            shutil.copy2(source, staging / part)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def uninstall_weapon_names(mods_dir: Path) -> None:
    target = mods_dir / WEAPON_NAMES_MOD
    if target.is_dir():
        shutil.rmtree(target)
