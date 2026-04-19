from __future__ import annotations

import re
from collections import Counter
from typing import Any, Final

from src.agent.state import GlossaryMatch
from src.services.weblate import WeblateClient

TAG_PATTERNS: Final = [
    r"<[^>]+>",
    r"%[dsiufxXpc]",
    r"\{[^}]*\}",
    r"\\[nt]",
    r"<XGParam:[^/]*/>",
]


def extract_tags(text: str) -> list[str]:
    tags: list[str] = []
    for pattern in TAG_PATTERNS:
        tags.extend(re.findall(pattern, text))
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
    return re.sub(r"<[^>]+>", "", text).strip()


def tokenize(text: str) -> set[str]:
    plain = re.sub(r"<[^>]+>", "", text)
    return {w.lower() for w in re.findall(r"[A-Za-z]{3,}", plain)}


def lookup_glossary(
    source: str, cache: dict[str, dict], limit: int = 10
) -> list[GlossaryMatch]:
    src_tokens = tokenize(source)
    if not src_tokens:
        return []

    scored: list[tuple[int, int, str, dict]] = []
    for src, info in cache.items():
        cache_tokens = tokenize(src)
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
    lang: str = "zh_Hans",
    nearby_range: int = 2,
) -> dict[str, Any]:
    search_query = strip_html(source_text) or source_text
    results = client.search_units(f'source:"{search_query}"', page_size=50)

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
    _page_size = 100
    page = 1
    while True:
        _count, units = client.list_units_page(
            best_slug, lang, page=page, page_size=_page_size
        )
        for u in units:
            pos = u["position"]
            if not (lo <= pos <= hi) or pos == target_pos:
                continue
            src = u["source"][0]
            if src in seen_sources or len(strip_html(src)) <= 2:
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
        if not units or len(units) < _page_size or units[-1]["position"] > hi:
            break
        page += 1

    nearby.sort(key=lambda x: x["pos"])
    return {
        "mod_component": best_slug,
        "translated_percent": round(best_pct, 1),
        "nearby": nearby,
    }
