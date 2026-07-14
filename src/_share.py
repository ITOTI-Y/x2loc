from pathlib import Path
from tempfile import mkdtemp
from typing import Final

LANG_EXT_MAP: Final[dict[str, str]] = {
    "int": "en",
    "chn": "zh_Hans",
    "cht": "zh_Hant",
    "deu": "de",
    "esn": "es",
    "fra": "fr",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "pol": "pl",
    "rus": "ru",
}

EXT_LANG_MAP: Final[dict[str, str]] = {v: k for k, v in LANG_EXT_MAP.items()}

TEMP_DIR: Path = Path(mkdtemp(prefix="x2loc_"))
TEMP_DIR = Path("data/_dev")


def make_glossary_context(source: str, category: str) -> str:
    """Build the Weblate unit context for a glossary term.

    Single source of truth for the `{source}::{category}` encoding shared
    by the CLI glossary upload paths and the agent's `fetch_empty` decode.
    """
    return f"{source}::{category}"


def glossary_context_category(context: str) -> str | None:
    """Inverse of `make_glossary_context`: extract the category segment."""
    return context.split("::")[-1] if "::" in context else None
