import re
from collections.abc import Iterator
from typing import Final

from src.models._share import PlaceholderType
from src.models.entry import EntrySchema
from src.models.file import LocalizationFile
from src.models.section import SectionSchema

COMMENT_PREFIX: Final[str] = ";"

SECTION_RE: Final[re.Pattern[str]] = re.compile(r"^\[(.+)\]\s*$")

ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(\+?)([\w.\- ]+(?:\[[^\]]+\])?)\s*=\s*(.*)"
)

ARRAY_INDEX_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]$")

PLACEHOLDER_PATTERNS: Final[list[tuple[re.Pattern[str], PlaceholderType]]] = [
    (re.compile(r"<XGParam:[^>]+/>"), PlaceholderType.XGPARAM),
    (re.compile(r"<Ability:[^>]+/>"), PlaceholderType.ABILITY),
    (re.compile(r"<\w+:[^>]+/>"), PlaceholderType.XML_VAR),
    (re.compile(r"<Bullet/>"), PlaceholderType.BULLET),
    (re.compile(r"<Heal/>"), PlaceholderType.HEAL),
    (re.compile(r"<br/>"), PlaceholderType.BR),
    (re.compile(r"<\w+/>"), PlaceholderType.XML_SELF_CLOSE),
    (
        re.compile(r"</?(?:font|h\d|br|div|span|p|b|i|u|em|strong)\b[^>]*>"),
        PlaceholderType.HTML,
    ),
    (re.compile(r"%[A-Za-z]+%"), PlaceholderType.PERCENT_WRAPPED),
    (re.compile(r"%[A-Za-z]"), PlaceholderType.PERCENT),
    (re.compile(r"\\n"), PlaceholderType.NEWLINE),
]


def make_compound_key(section_raw: str, key: str, ordinal: int | None = None) -> str:
    """Build a compound key from a section header raw text and an entry key.

    Format:
        "{section_raw}::{key}"          when ordinal is None
        "{section_raw}::{key}#{ordinal}" when ordinal is an int

    This function is the single source of truth for compound-key formatting.
    Both aligner (indexing) and converter (translation lookup) must go through
    it so that key shapes never drift between producer and consumer.
    """
    if ordinal is None:
        return f"{section_raw}::{key}"
    return f"{section_raw}::{key}#{ordinal}"


def iter_compound_keys_in_section(
    section: SectionSchema,
) -> Iterator[tuple[str, EntrySchema]]:
    """Yield `(compound_key, entry)` for every entry in one section.

    **Per-entry compound-key shape decision tree** (authoritative for
    aligner / converter / writeback):

    1. **Struct-append entry** (`+Key=(field=val, ...)`, `struct_fields`
       populated) → always indexed with `#N` ordinal suffix. These are
       true array-of-struct append operations where ordinal is semantic.
    2. **Repeated scalar key in same section** (same `key` appears >1
       times, counting non-append + scalar-append together) → indexed with
       `#N` ordinal. Preserves array-like `+.Credits="line 1"`,
       `+.Credits="line 2"` semantics.
    3. **Single-occurrence scalar** (appears exactly once, whether
       `key="..."` or `+key="..."`) → no suffix. This normalizes the
       common LW mod pattern where `.int` has `m_strUrgent="..."` and
       `.chn` has `+m_strUrgent="..."`; both sides collapse to the same
       compound key and the aligner pairs them correctly.

    Rationale: in UE3 loc files the `+` prefix is an array-append
    directive that is a no-op for single scalar writes but meaningful for
    arrays. We can't distinguish the property's type from the .int file
    alone, so we use OCCURRENCE COUNT as a proxy: one occurrence = scalar
    write, multiple occurrences = array. This heuristic aligns real
    XCOM 2 mod data correctly for both common cases.

    Ordinals are scoped to this single section.
    """
    # Pre-pass: count how many times each key appears as a scalar
    # (i.e. NOT a struct append). Multi-occurrence scalar keys get
    # ordinal suffixes; single-occurrence scalars do not.
    scalar_counts: dict[str, int] = {}
    for entry in section.entries:
        if entry.struct_fields is not None:
            continue  # struct appends are always indexed (rule 1)
        scalar_counts[entry.key] = scalar_counts.get(entry.key, 0) + 1

    append_counters: dict[str, int] = {}
    for entry in section.entries:
        if entry.is_append and entry.struct_fields is not None:
            # Rule 1: struct append → ordinal
            ordinal = append_counters.get(entry.key, 0)
            append_counters[entry.key] = ordinal + 1
            yield make_compound_key(section.header.raw, entry.key, ordinal), entry
        elif scalar_counts.get(entry.key, 0) > 1:
            # Rule 2: repeated scalar key → ordinal
            ordinal = append_counters.get(entry.key, 0)
            append_counters[entry.key] = ordinal + 1
            yield make_compound_key(section.header.raw, entry.key, ordinal), entry
        else:
            # Rule 3: single-occurrence scalar → no suffix
            yield make_compound_key(section.header.raw, entry.key), entry


def iter_compound_keys(
    file: LocalizationFile,
) -> Iterator[tuple[str, EntrySchema, SectionSchema]]:
    """Yield `(compound_key, entry, section)` triples in file order.

    Thin wrapper over `iter_compound_keys_in_section` that flattens across
    sections. Use this when the caller does not need the section boundary
    (e.g. aligner index, writeback translation counter). Empty sections
    contribute no elements.
    """
    for section in file.sections:
        for compound_key, entry in iter_compound_keys_in_section(section):
            yield compound_key, entry, section
