from __future__ import annotations

import asyncio
import hashlib
import unicodedata
from collections import defaultdict

from src.models.glossary import GlossaryTerm
from src.models.weblate import WeblateUnitDraftSchema, WeblateUnitSchema
from src.services.weblate import WEBLATE_STATE_TRANSLATED, AsyncWeblateClient


def normalize_term(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def term_pair(source: str, target: str) -> tuple[str, str]:
    return normalize_term(source), normalize_term(target)


def term_context(source: str, target: str) -> str:
    normalized = "\0".join(term_pair(source, target)).encode("utf-8")
    return f"auto::{hashlib.sha256(normalized).hexdigest()}"


def group_units(
    units: list[WeblateUnitSchema],
) -> dict[str, tuple[WeblateUnitSchema, ...]]:
    """Index units by NFC-normalized source, keeping every distinct target.

    Normalization is unconditional: an unnormalized key would make the exact
    match in `lookup_glossary_or_patterns` hit for one caller and miss for
    another on the same term.
    """
    grouped: dict[str, list[WeblateUnitSchema]] = defaultdict(list)
    for unit in units:
        source = normalize_term(unit.source)
        if source and normalize_term(unit.target):
            grouped[source].append(unit)
    return {source: tuple(values) for source, values in grouped.items()}


class CustomGlossaryWriter:
    """Appends newly mined term pairs to the writable custom glossary."""

    def __init__(
        self,
        client: AsyncWeblateClient,
        *,
        component_slug: str,
        target_lang: str,
    ) -> None:
        self._client = client
        self._component_slug = component_slug
        self._target_lang = target_lang
        self._lock = asyncio.Lock()

    async def write(self, terms: list[GlossaryTerm]) -> tuple[int, int]:
        """Create the pairs that are not present yet; return (added, skipped).

        The lock covers only the read-and-diff. Creating units is then done
        concurrently — the client's own semaphore bounds the request rate.
        """
        pairs = sorted(
            {
                term_pair(term.source, term.target)
                for term in terms
                if normalize_term(term.source)
                and normalize_term(term.target)
                and normalize_term(term.source) != normalize_term(term.target)
            }
        )
        if not pairs:
            return 0, 0

        async with self._lock:
            current = await self._client.list_units(
                self._component_slug, self._target_lang
            )
            existing = {
                term_pair(unit.source, unit.target)
                for unit in current
                if normalize_term(unit.source) and normalize_term(unit.target)
            }
            new_pairs = [pair for pair in pairs if pair not in existing]

        await asyncio.gather(
            *(
                self._client.create_unit(
                    self._component_slug,
                    self._target_lang,
                    WeblateUnitDraftSchema(
                        context=term_context(source, target),
                        source=source,
                        target=target,
                        state=WEBLATE_STATE_TRANSLATED,
                    ),
                )
                for source, target in new_pairs
            )
        )
        return len(new_pairs), len(pairs) - len(new_pairs)
