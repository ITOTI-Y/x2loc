import asyncio
import re
from collections import Counter
from typing import Final

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from src.agent._share import (
    DEFAULT_NEARBY_RANGE,
    MAX_CONTEXT_COMPONENTS,
    MAX_MATCHES_PER_COMPONENT,
)
from src.models.agent import (
    ComponentInfoSchema,
    PatternSchema,
)
from src.services.weblate import (
    AsyncWeblateClient,
    WeblateRequestParamsSchema,
    WeblateUnitSchema,
)

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


def lookup_glossary_or_patterns[T: (WeblateUnitSchema, PatternSchema, dict)](
    source: str, cache: dict[str, T], limit: int = 10
) -> list[T]:
    if source in cache:
        return [cache[source]]

    matched = process.extract(
        source,
        cache.keys(),
        scorer=WRatio,
        score_cutoff=65,
        limit=limit,
    )

    return [cache[m[0]] for m in matched]


async def collect_context_for_term(
    client: AsyncWeblateClient,
    input_unit: WeblateUnitSchema,
    nearby_range: int = DEFAULT_NEARBY_RANGE,
) -> list[ComponentInfoSchema]:
    async def _enrich(component: ComponentInfoSchema) -> ComponentInfoSchema:
        position_query = (
            f"position:[{component.position - nearby_range}"
            f" to {component.position + nearby_range}]"
        )
        nearby_page = await client.list_units_page(
            component_slug=component.slug,
            lang=component.lang,
            page=1,
            page_size=20,
            q=position_query,
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
        turl = u.translation
        unit_lang = u.language_code
        parts = turl.rstrip("/").split("/")
        slug = parts[-2]
        if len(parts) < 2:
            continue
        if slug.startswith("glossary"):
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

    seen_slugs: Counter[str] = Counter()
    picked: list[ComponentInfoSchema] = []
    for c in components:
        if seen_slugs[c.slug] >= MAX_MATCHES_PER_COMPONENT:
            continue
        seen_slugs[c.slug] += 1
        picked.append(c)
        if len(picked) >= MAX_CONTEXT_COMPONENTS:
            break

    enriched = await asyncio.gather(*[_enrich(c) for c in picked])

    return enriched
