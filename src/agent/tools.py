from __future__ import annotations

import re
from collections import Counter
from typing import Any, Final

from src.agent._share import (
    CONTEXT_MIN_SOURCE_LEN,
    CONTEXT_NEARBY_PAGE_SIZE,
    CONTEXT_SEARCH_PAGE_SIZE,
    DEFAULT_NEARBY_RANGE,
)
from src.agent.state import GlossaryMatch, SessionPattern
from src.services.weblate import WeblateClient

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

_HTML_TAG_RE: Final = re.compile(r"<[^>]+>")
_WORD_RE: Final = re.compile(r"[A-Za-z]{3,}")


def extract_tags(text: str) -> list[str]:
    tags: list[str] = []
    for pattern in TAG_PATTERNS:
        tags.extend(pattern.findall(text))
    return tags


def validate_tags(source: str, translation: str) -> tuple[bool, dict, dict]:
    src_tags = Counter(extract_tags(source))
    tgt_tags = Counter(extract_tags(translation))
    missing = {
        t: src_tags[t] - tgt_tags[t] for t in src_tags if src_tags[t] > tgt_tags[t]
    }
    extra = {
        t: tgt_tags[t] - src_tags[t] for t in tgt_tags if tgt_tags[t] > src_tags[t]
    }
    return (not missing and not extra), missing, extra


def strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text).strip()


def tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(strip_html(text))}


def make_glossary_entry(source: str, target: str, context: str) -> dict:
    """Build a glossary cache entry with its source tokens precomputed.

    `lookup_glossary` matches by token overlap; precomputing here avoids
    re-tokenizing every entry on every lookup.
    """
    return {"target": target, "context": context, "tokens": tokenize(source)}


def match_session_patterns(
    source: str, patterns: list[SessionPattern]
) -> list[SessionPattern]:
    """Collect session patterns whose src_pattern contains any word of `source`."""
    matched: list[SessionPattern] = []
    for single_word in source.split():
        matched.extend(
            p for p in patterns if single_word.lower() in p["src_pattern"].lower()
        )
    return matched


def lookup_glossary(
    source: str, cache: dict[str, dict], limit: int = 10
) -> list[GlossaryMatch]:
    src_tokens = tokenize(source)
    if not src_tokens:
        return []

    scored: list[tuple[int, int, str, dict]] = []
    for src, info in cache.items():
        cache_tokens = info.get("tokens") or tokenize(src)
        if not cache_tokens:
            continue
        overlap = src_tokens & cache_tokens
        if overlap:
            scored.append((len(overlap), len(cache_tokens), src, info))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        {"source": src, "target": info["target"], "context": info["context"]}
        for _, _, src, info in scored[:limit]
    ]


def collect_context_for_term(
    client: WeblateClient,
    source_text: str,
    lang: str,
    nearby_range: int = DEFAULT_NEARBY_RANGE,
) -> dict[str, Any]:
    search_query = strip_html(source_text) or source_text
    results = client.search_units(
        f'source:"{search_query}"', page_size=CONTEXT_SEARCH_PAGE_SIZE
    )

    components: dict[str, list[int]] = {}
    for u in results:
        turl = u.get("translation", "")
        parts = turl.rstrip("/").split("/")
        if len(parts) < 2:
            continue
        slug = parts[-2]
        unit_lang = parts[-1]
        if slug.startswith("glossary") or unit_lang != lang:
            continue
        if u["source"][0] != source_text:
            continue
        components.setdefault(slug, []).append(u["position"])

    if not components:
        return {"mod_component": None, "translated_percent": None, "nearby": []}

    best_slug, best_pct, best_positions = "", -1.0, []
    for slug, positions in components.items():
        info = client.get_translation(slug, lang)
        pct = info.get("translated_percent", 0.0)
        if pct > best_pct:
            best_slug, best_pct, best_positions = slug, pct, positions

    target_pos = best_positions[0]
    lo, hi = target_pos - nearby_range, target_pos + nearby_range

    nearby: list[dict] = []
    seen_sources: set[str] = set()
    # Positions are 1-based and page-ordered (the break test below relies on
    # that), so start directly at the page containing the window's low end
    # instead of scanning from page 1.
    page = max(1, (lo - 1) // CONTEXT_NEARBY_PAGE_SIZE + 1)
    while True:
        _count, units = client.list_units_page(
            best_slug, lang, page=page, page_size=CONTEXT_NEARBY_PAGE_SIZE
        )
        for u in units:
            pos = u["position"]
            if not (lo <= pos <= hi) or pos == target_pos:
                continue
            src = u["source"][0]
            if src in seen_sources or len(strip_html(src)) <= CONTEXT_MIN_SOURCE_LEN:
                continue
            seen_sources.add(src)
            nearby.append(
                {
                    "pos": pos,
                    "ctx": u["context"],
                    "src": src,
                    "tgt": u["target"][0] if u["target"][0] else None,
                }
            )
        if (
            not units
            or len(units) < CONTEXT_NEARBY_PAGE_SIZE
            or units[-1]["position"] > hi
        ):
            break
        page += 1

    nearby.sort(key=lambda x: x["pos"])
    return {
        "mod_component": best_slug,
        "translated_percent": round(best_pct, 1),
        "nearby": nearby,
    }
