from enum import StrEnum
from typing import Final, Literal, get_args

# Where an overlay's translation of a unit can come from. The source text is
# the implicit last resort and has no kind.
type SourceKind = Literal["manual", "upstream", "machine", "needs_editing", "shipped"]

SOURCE_KINDS: Final[tuple[SourceKind, ...]] = get_args(SourceKind.__value__)

# A player's own edit first; then what Weblate holds as translated, which
# passed a maintainer's process, ahead of this machine's unreviewed output.
DEFAULT_PRIORITY: Final[tuple[SourceKind, ...]] = (
    "manual",
    "upstream",
    "machine",
    "needs_editing",
    "shipped",
)


class StoreMode(StrEnum):
    """Where translations go; `AUTO` is Weblate when a token is set."""

    AUTO = "auto"
    WEBLATE = "weblate"
    LOCAL = "local"
