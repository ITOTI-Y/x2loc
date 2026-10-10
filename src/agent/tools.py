import re
from collections.abc import Mapping, Sequence
from functools import cache

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from src.models.agent import PatternSchema
from src.models.weblate import WeblateUnitSchema


def lookup_glossary(
    source: str,
    glossary: Mapping[str, Sequence[WeblateUnitSchema]],
    limit: int = 10,
) -> list[WeblateUnitSchema]:
    if source in glossary:
        return list(glossary[source])

    results: list[WeblateUnitSchema] = []
    matched_keys = _phrase_hits(source, glossary)
    for key in matched_keys:
        results.extend(glossary[key])
    if len(results) >= limit:
        return results[:limit]

    fuzzy = process.extract(
        source,
        glossary.keys(),
        scorer=WRatio,
        score_cutoff=65,
        limit=limit,
    )
    for match in fuzzy:
        if match[0] not in matched_keys:
            results.extend(glossary[match[0]])
    return results[:limit]


def match_patterns(
    source: str,
    patterns: Mapping[str, Sequence[PatternSchema]],
    limit: int = 5,
) -> list[PatternSchema]:
    """Templates whose literal words occur around a non-empty slot in `source`.

    Keys contain `{X}`, which neither word-boundary nor fuzzy glossary lookup
    can match, so each template compiles to its own regex. The most specific
    (longest literal) and best supported templates come first.
    """
    hits = [
        pattern
        for key, group in patterns.items()
        if _literals_present(key, source) and _template_regex(key).search(source)
        for pattern in group
    ]
    hits.sort(key=lambda p: (-len(p.src_pattern), -p.example_count))
    return hits[:limit]


def _literals_present(src_pattern: str, source: str) -> bool:
    """Cheap necessary condition for `_template_regex(src_pattern)` to match.

    The regex requires both literals verbatim, and its lazy middle group
    backtracks heavily on long sources; with thousands of templates the
    regex pass alone dominated prompt building.
    """
    prefix, suffix = _template_literals(src_pattern)
    return prefix in source and suffix in source


@cache
def _template_literals(src_pattern: str) -> tuple[str, str]:
    prefix, _, suffix = (part.strip() for part in src_pattern.partition("{X}"))
    return prefix, suffix


@cache
def _template_regex(src_pattern: str) -> re.Pattern[str]:
    prefix, suffix = _template_literals(src_pattern)
    parts: list[str] = []
    if prefix:
        parts.append(_edge(prefix, start=True) + re.escape(prefix) + r"\s+")
    parts.append(r"\S(?:.*?\S)?")
    if suffix:
        parts.append(r"\s+" + re.escape(suffix) + _edge(suffix, start=False))
    return re.compile("".join(parts))


def _edge(literal: str, *, start: bool) -> str:
    """Word boundary only where the literal itself begins/ends with a word char."""
    char = literal[0] if start else literal[-1]
    if not (char.isalnum() or char == "_"):
        return ""
    return r"(?<!\w)" if start else r"(?!\w)"


def _phrase_hits(
    source: str, glossary: Mapping[str, Sequence[WeblateUnitSchema]]
) -> set[str]:
    """Word-boundary hits of glossary keys inside the source text.

    Whole-string fuzzy matching never surfaces a short term inside a long
    sentence — "Sectoid" scores far below the cutoff against a 100-character
    quote — which starves the translator of exactly the official names the
    scorer later insists on. Possessive and plural suffixes are folded so
    "Sectoids" and "sectoid's" still hit the "Sectoid" entry.
    """
    words = re.findall(r"[a-z0-9']+", source.lower())
    phrases: set[str] = set()
    for word in words:
        phrases.add(word)
        if word.endswith("'s"):
            phrases.add(word[:-2])
        elif word.endswith("s"):
            phrases.add(word[:-1])
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrases.add(" ".join(words[i : i + n]))
    return {key for key in glossary if len(key) >= 3 and key.lower() in phrases}
