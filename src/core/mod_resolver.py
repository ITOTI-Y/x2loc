"""Resolve a loc-file directory into a canonical mod identity.

This module answers one question: "which mod does this .int file belong to,
and what Weblate namespace should it live under?"

Resolution flow:

1. **Walk-up**: starting from a user-supplied directory (usually the one
   containing .int/.chn files), walk up parent directories looking for a
   single `.XComMod` manifest. The walk is bounded by a caller-supplied
   sandbox root (a hard boundary) and a secondary level cap.
2. **Manifest parse**: extract the first non-empty `Title` and the
   `publishedFileId` fields from the manifest. The `Title` key may appear
   twice in one file (lowercase empty + capitalized real value — an
   artifact of the mod template), so we pick the first non-empty match.
3. **Steam ID resolution**: prefer an explicit override (from the CLI or
   the uploaded zip name); then a `PublishedFileId.ID` sidecar; then a
   numeric mod-root directory name; then the manifest's `publishedFileId`
   field if non-zero. If none of these yield a value, we degrade to a
   `local-{title_slug}` namespace and let the caller decide whether that
   is acceptable.
4. **Slug**: transliterate the title with `unidecode` so CJK titles
   survive as readable ASCII, then lowercase + hyphen + truncate.

The sandbox-root constraint is the hard stop: in production, each job
runs in its own extraction temp dir, and walk-up must never cross that
dir. Symlinks are deliberately NOT followed (via `os.path.normpath` over
`Path.resolve()`) so that local experiments using symlinks into `data/`
behave the same way a real zip extraction would.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Final

from loguru import logger
from unidecode import unidecode

from src.models.mod import BASE_GAME_NAMESPACE, ModInfoSchema

MAX_WALKUP_LEVELS: Final[int] = 5
XCOMMOD_SUFFIX: Final[str] = ".xcommod"
MOD_SECTION_HEADER: Final[str] = "[mod]"
PUBLISHED_FILE_ID_SIDECAR: Final[str] = "PublishedFileId.ID"
SLUG_MAX_LENGTH: Final[int] = 40

_SLUG_CLEAN_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


class ModResolveError(Exception):
    """Raised when a mod identity cannot be resolved.

    Causes include: walk-up exited the sandbox, no .XComMod found within
    the allowed range, multiple .XComMod manifests in one directory, or
    the manifest has no non-empty Title field.
    """


def resolve_mod(
    start: Path,
    sandbox_root: Path,
    steam_id_override: str | None = None,
    max_levels: int = MAX_WALKUP_LEVELS,
) -> ModInfoSchema:
    """Resolve a mod's canonical identity from a starting directory.

    Args:
        start: Directory inside the mod (usually a loc dir). Walk-up
            begins here and proceeds toward `sandbox_root`.
        sandbox_root: Absolute boundary for the walk-up. Must be an
            ancestor of `start` after lexical normalization; raises
            `ModResolveError` otherwise.
        steam_id_override: If the caller already knows the Steam Workshop
            ID (e.g. extracted from an uploaded zip filename or a
            `--steam-id` CLI flag), pass it here to bypass filesystem
            inference — filesystem inference can't distinguish a zip
            extracted into a random temp dir from an unpublished local
            mod.
        max_levels: Secondary safety cap on walk-up distance. Defaults
            to `MAX_WALKUP_LEVELS` (5); real-world data tops out at 3.

    Returns:
        A fully populated `ModInfoSchema`.

    Raises:
        ModResolveError: on any resolution failure.
    """
    mod_root = find_mod_root(start, sandbox_root, max_levels=max_levels)
    xcommod_path = _locate_xcommod(mod_root)
    title = read_xcommod_title(xcommod_path)
    steam_id = steam_id_override or _derive_steam_id(mod_root, xcommod_path)
    title_slug = weblate_slug(title)

    namespace = f"{steam_id}-{title_slug}" if steam_id else f"local-{title_slug}"
    if not steam_id:
        logger.warning(
            f"No Steam Workshop ID for mod '{title}' at {mod_root}; "
            f"falling back to namespace '{namespace}'. Pass --steam-id "
            "to anchor this mod to its Workshop entry."
        )

    return ModInfoSchema(
        namespace=namespace,
        steam_id=steam_id,
        mod_title=title,
        mod_root=mod_root,
        xcommod_path=xcommod_path,
    )


def find_mod_root(
    start: Path, sandbox_root: Path, max_levels: int = MAX_WALKUP_LEVELS
) -> Path:
    """Walk up from `start` looking for the directory containing `.XComMod`.

    Args:
        start: Directory to begin searching from. If `start` is a file,
            its parent is used as the effective start.
        sandbox_root: Hard boundary — walk-up stops here even if no
            manifest has been found.
        max_levels: Secondary cap on parent hops.

    Returns:
        Absolute, lexically-normalized path to the directory holding the
        `.XComMod` manifest.

    Raises:
        ModResolveError: if `start` is outside `sandbox_root`, no
            manifest was found within range, or more than one manifest
            coexists in a single directory.
    """
    start_norm = _lexical_normalize(start)
    sandbox_norm = _lexical_normalize(sandbox_root)

    try:
        start_norm.relative_to(sandbox_norm)
    except ValueError as e:
        raise ModResolveError(
            f"Start path {start_norm} is not inside sandbox root {sandbox_norm}"
        ) from e

    cur = start_norm if start_norm.is_dir() else start_norm.parent
    for _ in range(max_levels + 1):  # +1 so `start` itself is inspected
        matches = sorted(
            p
            for p in cur.iterdir()
            if p.is_file() and p.name.lower().endswith(XCOMMOD_SUFFIX)
        )
        if len(matches) > 1:
            raise ModResolveError(
                f"Multiple .XComMod manifests in {cur}: {[m.name for m in matches]}"
            )
        if matches:
            return cur
        if cur == sandbox_norm:
            break
        parent = cur.parent
        if parent == cur:  # filesystem root
            break
        cur = parent

    raise ModResolveError(
        f"No .XComMod found walking up from {start_norm} "
        f"(stopped at {cur}, max_levels={max_levels})"
    )


def read_xcommod_title(xcommod_path: Path) -> str:
    """Return the first non-empty `Title` value from a `.XComMod` file.

    The mod template sometimes leaves an empty lowercase `title=` line
    before the real `Title=Whatever` assignment. Plain `ConfigParser`
    loses the second value to the first on duplicate keys, so we scan
    the file line by line and take the first match whose value is
    non-empty.

    Raises:
        ModResolveError: if no non-empty Title is found.
    """
    for key, value in _iter_mod_section(xcommod_path):
        if key == "title" and value:
            return value
    raise ModResolveError(f"No non-empty Title in {xcommod_path}")


def weblate_slug(text: str, max_len: int = SLUG_MAX_LENGTH) -> str:
    """Convert a free-form string into a Weblate-safe lowercase slug.

    Non-ASCII input (e.g. CJK mod titles) is transliterated via
    `unidecode` so `中国武器包` → `zhong-guo-wu-qi-bao` instead of
    collapsing to empty.

    Args:
        text: Source string (mod title, etc.).
        max_len: Maximum slug length. Longer slugs are truncated at a
            word boundary where possible.

    Returns:
        A hyphen-separated lowercase ASCII slug. Never empty; falls
        back to `"unnamed"` if the input yields no usable characters.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = unidecode(normalized)
    collapsed = _SLUG_CLEAN_RE.sub("-", ascii_text.lower()).strip("-")
    if not collapsed:
        return "unnamed"
    truncated = collapsed[:max_len].rstrip("-")
    return truncated or "unnamed"


def _lexical_normalize(p: Path) -> Path:
    """Absolute + `..` collapsed, WITHOUT following symlinks.

    `Path.absolute()` preserves `..`, so `temp/../data` stays that way
    and defeats containment checks. `Path.resolve()` would fix that but
    also follows symlinks, corrupting local experiments that symlink
    mods into the sandbox. `os.path.normpath` is pure-lexical and does
    exactly what is needed.
    """
    return Path(os.path.normpath(str(p.absolute())))


def _locate_xcommod(mod_root: Path) -> Path:
    """Return the single `.XComMod` file in `mod_root`.

    Caller must have already validated that exactly one manifest exists
    via `find_mod_root`; this is a strict accessor that fails loudly on
    surprise multi-manifest layouts.
    """
    matches = sorted(
        p
        for p in mod_root.iterdir()
        if p.is_file() and p.name.lower().endswith(XCOMMOD_SUFFIX)
    )
    if not matches:
        raise ModResolveError(f"No .XComMod in {mod_root}")
    if len(matches) > 1:
        raise ModResolveError(
            f"Multiple .XComMod files in {mod_root}: {[m.name for m in matches]}"
        )
    return matches[0]


def _iter_mod_section(xcommod_path: Path):
    """Yield `(key_lowercased, value_stripped)` pairs from the `[mod]`
    section of a `.XComMod` file.

    Bypasses `configparser` because the template duplicate-key trap
    (empty `title=` followed by real `Title=`) is silently resolved
    last-write-wins there, which drops the real value.
    """
    text = xcommod_path.read_text(encoding="utf-8-sig", errors="replace")
    in_mod_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_mod_section = line.lower() == MOD_SECTION_HEADER
            continue
        if not in_mod_section or "=" not in line:
            continue
        k, _, v = line.partition("=")
        yield k.strip().lower(), v.strip()


def _read_published_file_id_sidecar(mod_root: Path) -> str | None:
    """Read `PublishedFileId.ID` if present — used by source-package mods
    (e.g. LWOTC) where the `.XComMod` carries `publishedFileId=0` and the
    real ID lives in a separate file next to it.
    """
    sidecar = mod_root / PUBLISHED_FILE_ID_SIDECAR
    if not sidecar.is_file():
        return None
    text = sidecar.read_text(encoding="utf-8-sig", errors="replace").strip()
    if text and text.isdigit() and text != "0":
        return text
    return None


def _read_xcommod_published_file_id(xcommod_path: Path) -> str | None:
    """Read the `publishedFileId` field from a `.XComMod` file.

    Returns None when the field is missing, empty, or the placeholder
    `0` — about 25% of real-world manifests carry `publishedFileId=0`
    because mod authors leave it blank pre-upload and never update it.
    """
    for key, value in _iter_mod_section(xcommod_path):
        if key == "publishedfileid" and value and value != "0":
            return value
    return None


def _derive_steam_id(mod_root: Path, xcommod_path: Path) -> str | None:
    """Resolve a mod's Steam Workshop ID using the three-tier fallback.

    Priority order (most to least reliable):
        1. `PublishedFileId.ID` sidecar file next to the manifest.
        2. Mod root directory name, if it is all-digits (the Steam
           Workshop download layout).
        3. Manifest `publishedFileId` field (only if non-zero — it is
           a placeholder in ~25% of real-world mods).

    Returns None when all three fail. The caller is responsible for
    degrading to `local-{slug}` or requiring `--steam-id`.
    """
    sidecar = _read_published_file_id_sidecar(mod_root)
    if sidecar:
        return sidecar

    dir_name = mod_root.name
    if dir_name.isdigit() and dir_name != "0":
        return dir_name

    return _read_xcommod_published_file_id(xcommod_path)


__all__ = [
    "BASE_GAME_NAMESPACE",
    "MAX_WALKUP_LEVELS",
    "ModResolveError",
    "find_mod_root",
    "read_xcommod_title",
    "resolve_mod",
    "weblate_slug",
]
