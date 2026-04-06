import re
from pathlib import Path
from typing import Final

from src.models._share import PlaceholderType

COMMENT_PREFIX: Final[str] = ";"

SECTION_RE: Final[re.Pattern[str]] = re.compile(r"^\[(.+)\]\s*$")

ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(\+?)([\w.\- ]+(?:\[[^\]]+\])?)\s*=\s*(.*)"
)

ARRAY_INDEX_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]$")

PLACEHOLDER_PATTERNS: Final[list[tuple[re.Pattern[str], PlaceholderType]]] = [
    (re.compile(r"<XGParam:[^>]+/>"), PlaceholderType.XGPARAM),
    (re.compile(r"<Ability:[^>]+/>"), PlaceholderType.ABILITY),
    (re.compile(r"<Bullet/>"), PlaceholderType.BULLET),
    (re.compile(r"<Heal/>"), PlaceholderType.HEAL),
    (re.compile(r"<br/>"), PlaceholderType.BR),
    (re.compile(r"<font[^>]*>"), PlaceholderType.HTML),
    (re.compile(r"</font>"), PlaceholderType.HTML),
    (re.compile(r"%[A-Za-z]+%"), PlaceholderType.PERCENT_WRAPPED),
    (re.compile(r"%[A-Za-z]"), PlaceholderType.PERCENT),
    (re.compile(r"\\n"), PlaceholderType.NEWLINE),
]


class ParseError(Exception):
    """Raised when a localization file has structural issues."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")
