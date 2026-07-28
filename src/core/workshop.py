import hashlib
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import parse_qs, urlparse

from src.models.workshop import (
    SOURCE_SUFFIX,
    TARGET_SUFFIX,
    LocalizationAssetSchema,
    WorkshopItemSchema,
    WorkshopLimitsSchema,
)

STEAM_HOSTS: Final[frozenset[str]] = frozenset(
    {"steamcommunity.com", "www.steamcommunity.com"}
)
LOCALIZATION_DIR: Final[str] = "localization"


class WorkshopInputError(ValueError):
    """Workshop URL or downloaded content violates an input constraint."""


def parse_workshop_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in STEAM_HOSTS:
        raise WorkshopInputError("only HTTPS steamcommunity.com URLs are accepted")
    try:
        port = parsed.port
    except ValueError as exc:
        raise WorkshopInputError("Workshop URL contains an invalid port") from exc
    if parsed.username or parsed.password or port not in {None, 443}:
        raise WorkshopInputError("Workshop URL contains unsupported authority data")
    if parsed.path.rstrip("/") != "/sharedfiles/filedetails":
        raise WorkshopInputError("URL must point to one Workshop item")
    values = parse_qs(parsed.query, keep_blank_values=True).get("id", [])
    if len(values) != 1 or not values[0].isdecimal() or values[0] == "0":
        raise WorkshopInputError("URL must contain one non-zero numeric id")
    return values[0]


def component_slug(workshop_id: str, relative_source_path: PurePosixPath) -> str:
    encoded = relative_source_path.as_posix().encode("utf-8")
    return f"mod-{workshop_id}-{hashlib.sha256(encoded).hexdigest()[:12]}"


def scan_mod_tree(mod_root: Path, limits: WorkshopLimitsSchema) -> list[Path]:
    root = mod_root.resolve(strict=True)
    files: list[Path] = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise WorkshopInputError("Workshop content contains a symbolic link")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise WorkshopInputError("Workshop content escaped the mod root") from exc
        size = resolved.stat().st_size
        if resolved.suffix.casefold() == SOURCE_SUFFIX and size > (
            limits.max_loc_file_bytes
        ):
            raise WorkshopInputError("localization file exceeds size limit")
        total_bytes += size
        files.append(resolved)
        if len(files) > limits.max_file_count:
            raise WorkshopInputError("Workshop content exceeds the file count limit")
        if total_bytes > limits.max_total_bytes:
            raise WorkshopInputError("Workshop content exceeds the total size limit")
    return files


def discover_localization_assets(
    item: WorkshopItemSchema,
) -> list[LocalizationAssetSchema]:
    root = item.mod_root.resolve(strict=True)
    existing = set(item.files)
    sources = [
        path
        for path in item.files
        if path.suffix.casefold() == SOURCE_SUFFIX
        and any(
            part.casefold() == LOCALIZATION_DIR
            for part in path.relative_to(root).parts[:-1]
        )
    ]
    if not sources:
        raise WorkshopInputError(
            f"no {SOURCE_SUFFIX} file found below Localization directories"
        )

    collections: dict[str, PurePosixPath] = {}
    assets: list[LocalizationAssetSchema] = []
    for source in sources:
        relative_source = PurePosixPath(source.relative_to(root).as_posix())
        relative_target = relative_source.with_suffix(TARGET_SUFFIX)
        target = source.with_suffix(TARGET_SUFFIX)
        asset = LocalizationAssetSchema(
            source_path=source,
            existing_target_path=target if target in existing else None,
            relative_source_path=relative_source,
            relative_target_path=relative_target,
            component_slug=component_slug(item.workshop_id, relative_source),
        )
        previous = collections.get(asset.windows_collision_key)
        if previous is not None:
            raise WorkshopInputError(
                f"Windows output collision: {previous} and {relative_target}"
            )
        collections[asset.windows_collision_key] = relative_target
        assets.append(asset)
    return assets
