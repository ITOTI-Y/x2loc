"""Placeholder validation shared by the agent, the CLI review and the artifact gate.

The parser's `PLACEHOLDER_PATTERNS` (core/_share.py) classifies placeholders
for corpus metadata; this module is the single authority on whether a
translation preserved every tag of its source.
"""

import re
from collections import Counter
from typing import Final

TAG_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(p)
    for p in (
        r"<[^>]+>",
        r"%[dsiufxXpc]",
        r"\{[^}]*\}",
        r"\\[nt]",
        r"<XGParam:[^/]*/>",
    )
]


# An opening tag whose quoted attributes run straight into text, its `>`
# lost: `<font color='#df07b7'Smart` is read as `<font color='#df07b7'>Smart`.
# Only a quoted value followed by a character that cannot continue the tag
# qualifies, so a well-formed tag is never touched.
_UNCLOSED_OPENER: Final = re.compile(
    r"""<([A-Za-z][\w-]*)((?:\s+[\w-]+\s*=\s*(?:'[^'<>]*'|"[^"<>]*"))+)(?=[^\s>/])"""
)


def repair_markup(text: str) -> str:
    """Restore the `>` an opening tag visibly lost; other text is unchanged."""
    return _UNCLOSED_OPENER.sub(r"<\1\2>", text)


def is_malformed(text: str) -> bool:
    """Whether a `<...>` tag swallowed another tag's `<`, i.e. lost its `>`."""
    return any("<" in tag[1:] for tag in TAG_PATTERNS[0].findall(text))


def extract_tags(text: str) -> list[str]:
    tags: list[str] = []
    for pattern in TAG_PATTERNS:
        tags.extend(pattern.findall(text))
    return tags


def _intended(tag: str) -> str:
    """The well-formed tag a malformed one ends in: `<<Bullet/>` -> `<Bullet/>`."""
    return tag[tag.rindex("<") :] if tag.startswith("<") else tag


def validate_tags(source: str, translation: str) -> tuple[bool, dict, dict]:
    """Whether `translation` keeps every tag of `source`.

    The source is compared after `repair_markup`, so a translation that
    closes a tag the mod author left open is valid. When the source stays
    malformed even then, a translation that reproduces its tags verbatim is
    valid, and so is a well-formed one (a human's, as a rule, is the
    authority there) that keeps every tag the source evidently meant; it
    may add tags, such as an opener the source lost entirely.
    """
    source = repair_markup(source)
    src_tags = Counter(extract_tags(source))
    tgt_tags = Counter(extract_tags(translation))
    missing = {
        t: src_tags[t] - tgt_tags[t] for t in src_tags if src_tags[t] > tgt_tags[t]
    }
    extra = {
        t: tgt_tags[t] - src_tags[t] for t in tgt_tags if tgt_tags[t] > src_tags[t]
    }
    if (missing or extra) and is_malformed(source) and not is_malformed(translation):
        intended = Counter(_intended(tag) for tag in src_tags.elements())
        missing = {t: n - tgt_tags[t] for t, n in intended.items() if n > tgt_tags[t]}
        extra = {}
    return (not missing and not extra), missing, extra
