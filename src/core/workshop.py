import hashlib
from pathlib import Path, PurePosixPath
from typing import Final
from urllib.parse import parse_qs, urlparse

from src.models.workshop import (
    SOURCE_SUFFIX,
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
    for candidate in sorted(root.glob("**/*.int")):
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
