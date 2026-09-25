from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict

from src.models.weblate import WeblateUnitSchema


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
