"""The Workshop mods on this machine, as the app lists them.

Listing reads each mod's manifest and file names but parses no file, so it
stays fast enough to repeat; translating parses them (see `src.cli.local`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.core.mod_resolver import ModResolveError
from src.core.overlay_mod import overlay_mod_name
from src.core.workshop import (
    WorkshopInputError,
    discover_localization_assets,
    load_workshop_item,
)
from src.models.weblate import WeblateTranslationStatsSchema
from src.models.workshop import WorkshopLimitsSchema

type ModProblem = Literal["no_localization", "invalid_mod"]


@dataclass(frozen=True)
class ModEntry:
    workshop_id: str
    title: str
    # Weblate component of each localization file.
    slugs: tuple[str, ...]
    overlay_installed: bool
    problem: ModProblem | None = None


@dataclass(frozen=True)
class Coverage:
    translated: int
    total: int
    # Components of the mod Weblate holds; the rest are not synced yet.
    components_found: int


def scan_mods(
    workshop_dir: Path, mods_dir: Path | None, limits: WorkshopLimitsSchema
) -> list[ModEntry]:
    entries: list[ModEntry] = []
    for mod_root in sorted(workshop_dir.iterdir()):
        if not (mod_root.is_dir() and mod_root.name.isdecimal()):
            continue
        workshop_id = mod_root.name
        installed = (
            mods_dir is not None and (mods_dir / overlay_mod_name(workshop_id)).is_dir()
        )
        try:
            item = load_workshop_item(mod_root, workshop_id, limits)
        except (WorkshopInputError, ModResolveError):
            entries.append(
                ModEntry(workshop_id, workshop_id, (), installed, "invalid_mod")
            )
            continue
        title = item.mod_info.mod_title
        try:
            assets = discover_localization_assets(item)
        except WorkshopInputError:
            entries.append(
                ModEntry(workshop_id, title, (), installed, "no_localization")
            )
            continue
        slugs = tuple(asset.component_slug for asset in assets)
        entries.append(ModEntry(workshop_id, title, slugs, installed))
    return entries


def coverage(
    entry: ModEntry, stats: dict[str, WeblateTranslationStatsSchema]
) -> Coverage:
    found = [stats[slug] for slug in entry.slugs if slug in stats]
    return Coverage(
        translated=sum(s.translated for s in found),
        total=sum(s.total for s in found),
        components_found=len(found),
    )
