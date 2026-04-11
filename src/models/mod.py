from pathlib import Path
from typing import Final

from pydantic import Field

from src.models._share import BaseSchema

BASE_GAME_NAMESPACE: Final[str] = "base-xcom2-wotc"
BASE_GAME_TITLE: Final[str] = "XCOM 2 War of the Chosen"


class ModInfoSchema(BaseSchema):
    """Resolved identity of a single XCOM 2 mod (or the base game).

    The namespace is the stable, collision-free identifier we use for
    Weblate component slugs, corpus output directories, and every
    cross-reference in the pipeline. It is built from a Steam Workshop ID
    (when one is known) plus a transliterated title slug, so a new
    maintainer seeing `1122837889-more-traits` can immediately look the
    mod up on the Workshop.
    """

    namespace: str = Field(
        ...,
        description=(
            "Weblate/slug-safe identity. Shape: `{steam_id}-{title_slug}`, "
            "or `local-{title_slug}` when no Steam ID is known, or the fixed "
            "constant `base-xcom2-wotc` for the base game."
        ),
    )
    steam_id: str | None = Field(
        default=None,
        description=(
            "Steam Workshop published file ID. None for local/unpublished "
            "mods and for the base game."
        ),
    )
    mod_title: str = Field(
        ...,
        description=(
            "Human-readable mod name from the .XComMod Title field, or the "
            "base-game title constant."
        ),
    )
    mod_root: Path | None = Field(
        default=None,
        description=(
            "Absolute path to the mod root directory (where .XComMod lives). "
            "None for the base game, which has no manifest."
        ),
    )
    xcommod_path: Path | None = Field(
        default=None,
        description=(
            "Path to the resolved .XComMod manifest file, or None for the base game."
        ),
    )

    @classmethod
    def base_game(cls) -> "ModInfoSchema":
        """Return the canonical base-game ModInfoSchema.

        Used by `align-dir --base-game` to bypass mod_resolver and produce
        a corpus tagged with the fixed base-game namespace.
        """
        return cls(
            namespace=BASE_GAME_NAMESPACE,
            steam_id=None,
            mod_title=BASE_GAME_TITLE,
            mod_root=None,
            xcommod_path=None,
        )
