import asyncio
import re
from collections import Counter
from typing import Final

from rapidfuzz import process
from rapidfuzz.fuzz import WRatio

from src.agent._share import (
    DEFAULT_NEARBY_RANGE,
)
from src.models.agent import ComponentInfoSchema, PatternSchema
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


def lookup_glossary_or_patterns[T: (WeblateUnitSchema, PatternSchema)](
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
        info, nearby_page = await asyncio.gather(
            client.get_translation(component.slug, component.lang),
            client.list_units_page(
                component_slug=component.slug,
                lang=component.lang,
                page=1,
                page_size=20,
                q=position_query,
            ),
        )
        component.translated_percent = info.translated_percent
        component.nearby = [u for u in nearby_page.units if component.key in u.context]
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

    components = await asyncio.gather(*[_enrich(c) for c in components])

    return components
