from __future__ import annotations

import re
from collections import Counter
from typing import Final

from src.agent._share import (
    DEFAULT_NEARBY_RANGE,
)
from src.agent.state import GlossaryMatch, SessionPattern
from src.models.agent import ComponentInfoSchema
from src.services.weblate import AsyncWeblateClient, WeblateRequestParamsSchema

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


async def collect_context_for_term(
    client: AsyncWeblateClient,
    source_text: str,
    lang: str,
    nearby_range: int = DEFAULT_NEARBY_RANGE,
) -> list[ComponentInfoSchema]:
    search_query = strip_html(source_text) or source_text
    units = await client.search_units(
        WeblateRequestParamsSchema(
            page_size=50,
            q=f'source:"{search_query}"',
        )
    )

    components: list[ComponentInfoSchema] = []
    for u in units:
        turl = u.translation
        unit_lang = u.language_code
        parts = turl.rstrip("/").split("/")
        slug = parts[-2]
        if len(parts) < 2:
            continue
        if slug.startswith("glossary") or unit_lang != lang:
            continue
        if u.source != source_text:
            continue
        components.append(
            ComponentInfoSchema(
                unit=u,
                key=u.context.split("::")[0],
                slug=slug,
                lang=unit_lang,
                position=u.position,
                nearby=[],
            )
        )

    if not components:
        return []

    for component in components:
        info = await client.get_translation(component.slug, component.lang)
        component.translated_percent = info.translated_percent
        pos = component.position
        component.nearby = [
            u
            for u in await client.list_units(
                component.slug,
                component.lang,
                q=f"position:[{pos - nearby_range} to {pos + nearby_range}]",
            )
            if component.key in u.context
        ]

    return components
