import asyncio
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from src.agent._share import (
    DEFAULT_NEARBY_RANGE,
    MAX_CONTEXT_COMPONENTS,
    MAX_MATCHES_PER_COMPONENT,
)
from src.models.agent import ComponentInfoSchema, PatternSchema
from src.models.weblate import WeblateRequestParamsSchema, WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient

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


def lookup_glossary_or_patterns[T: (WeblateUnitSchema, PatternSchema)](
    source: str,
    cache: Mapping[str, Sequence[T]],
    limit: int = 10,
) -> list[T]:
    if source in cache:
        return list(cache[source])

    results: list[T] = []
    matched_keys = _phrase_hits(source, cache)
    for key in matched_keys:
        results.extend(cache[key])
    if len(results) >= limit:
        return results[:limit]

    fuzzy = process.extract(
        source,
        cache.keys(),
        scorer=WRatio,
        score_cutoff=65,
        limit=limit,
    )
    for match in fuzzy:
        if match[0] not in matched_keys:
            results.extend(cache[match[0]])
    return results[:limit]


def _phrase_hits[T](source: str, cache: Mapping[str, Sequence[T]]) -> set[str]:
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
    return {key for key in cache if len(key) >= 3 and key.lower() in phrases}


async def collect_context_for_term(
    client: AsyncWeblateClient,
    input_unit: WeblateUnitSchema,
    nearby_range: int = DEFAULT_NEARBY_RANGE,
    exclude_slug: str = "",
) -> list[ComponentInfoSchema]:
    """Collect cross-component context for one source string.

    `exclude_slug` keeps the component being translated out of its own
    context: its nearby units are sibling fields of the same template
    (title next to description), and feeding those back misleads both the
    translator and the scorer into swapping field contents.
    """

    async def _enrich(component: ComponentInfoSchema) -> ComponentInfoSchema:
        nearby_page = await client.list_units_page(
            component.slug,
            component.lang,
            WeblateRequestParamsSchema(
                page_size=20,
                q=(
                    f"position:[{component.position - nearby_range}"
                    f" to {component.position + nearby_range}]"
                ),
            ),
        )
        component.nearby = [
            u for u in nearby_page.results if component.key in u.context
        ]
        return component

    search_query = strip_html(input_unit.source) or input_unit.source
    units = await client.search_units(
        WeblateRequestParamsSchema(
            page_size=20,
            q=(
                f'source:="{search_query}"'
                f" AND language:{input_unit.language_code}"
                f" AND project:{client.config.project_slug}"
            ),
        )
    )

    components: list[ComponentInfoSchema] = []
    for u in units:
        parts = u.translation.rstrip("/").split("/")
        if len(parts) < 2:
            continue
        slug = parts[-2]
        if slug.startswith("glossary") or slug == exclude_slug:
            continue
        components.append(
            ComponentInfoSchema(
                unit=u,
                key=u.context.split("::")[0],
                slug=slug,
                lang=u.language_code,
                position=u.position,
                nearby=[],
            )
        )

    if not components:
        return []

    seen_slugs: Counter[str] = Counter()
    picked: list[ComponentInfoSchema] = []
    for c in components:
        if seen_slugs[c.slug] >= MAX_MATCHES_PER_COMPONENT:
            continue
        seen_slugs[c.slug] += 1
        picked.append(c)
        if len(picked) >= MAX_CONTEXT_COMPONENTS:
            break

    return list(await asyncio.gather(*[_enrich(c) for c in picked]))
