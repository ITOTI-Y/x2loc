from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import reduce
from os.path import commonprefix
from typing import TypedDict

from loguru import logger

from src.agent._share import (
    PATTERN_LITERAL_MAJORITY,
    PATTERN_MAX_EXAMPLES,
    PATTERN_MAX_SOURCE_WORDS,
    PATTERN_MIN_EXAMPLES,
)
from src.core.glossary import normalize_term
from src.core.placeholders import validate_tags
from src.models.agent import (
    NewAgentStateSchema,
    PatternExampleSchema,
    PatternSchema,
)
from src.models.weblate import WeblateUnitSchema

type _TemplateKey = tuple[tuple[str, ...], tuple[str, ...]]


class PatternExtractorOutputSchema(TypedDict):
    approved_pairs: dict[str, str]
    patterns: dict[str, tuple[PatternSchema, ...]]


def mine_glossary_patterns(
    base: Mapping[str, Sequence[WeblateUnitSchema]],
    mods: Mapping[str, Sequence[WeblateUnitSchema]],
) -> dict[str, tuple[PatternSchema, ...]]:
    """Derive translation templates from the Weblate glossaries.

    Mining runs three times — base alone, mods alone, and both together —
    because two glossaries that translate one template differently leave no
    shared target affix in the combined run, while a template with two
    examples in each glossary only reaches the minimum when combined. On
    conflicting templates the base (official) glossary wins, then the
    combined run, then mods.
    """
    base_pairs, mod_pairs = _first_targets(base), _first_targets(mods)
    translations = literal_translations(base, mods)
    patterns: dict[str, tuple[PatternSchema, ...]] = {}
    for pairs in (mod_pairs, mod_pairs | base_pairs, base_pairs):
        found = _detect_patterns(pairs, translations)
        patterns |= {key: (p,) for key, p in found.items()}
    return patterns


def literal_translations(
    base: Mapping[str, Sequence[WeblateUnitSchema]],
    mods: Mapping[str, Sequence[WeblateUnitSchema]],
) -> dict[str, str]:
    """Case-folded source -> target lookup for template literals; base wins."""
    return {
        source.casefold(): target
        for source, target in (_first_targets(mods) | _first_targets(base)).items()
    }


def _first_targets(
    glossary: Mapping[str, Sequence[WeblateUnitSchema]],
) -> dict[str, str]:
    """One target per source; the smallest keeps mining deterministic."""
    return {
        source: min(normalize_term(unit.target) for unit in units)
        for source, units in glossary.items()
    }


def pattern_extractor(state: NewAgentStateSchema) -> PatternExtractorOutputSchema:
    """Add templates mined from this session's human-approved pairs.

    They live in graph state only; the durable source is the glossaries.
    """
    pairs = _collect_pairs(state)
    patterns = dict(state.patterns)

    if len(pairs) >= PATTERN_MIN_EXAMPLES:
        translations = literal_translations(state.base_glossary, state.mods_glossary)
        for src_pattern, mined in _detect_patterns(pairs, translations).items():
            current = patterns.get(src_pattern)
            if current is not None and current[0].example_count >= mined.example_count:
                continue
            patterns[src_pattern] = (mined,)
            if current is None:
                logger.info(
                    f'[PATTERN] "{mined.src_pattern}" → "{mined.tgt_pattern}"'
                    f" ({mined.example_count} examples)"
                )
    return {"approved_pairs": pairs, "patterns": patterns}


def _collect_pairs(state: NewAgentStateSchema) -> dict[str, str]:
    pairs = dict(state.approved_pairs)
    units = {u.id: u for u in state.scores}
    for decision in state.decisions:
        unit = units.get(decision.unit_id)
        if unit is None or decision.action == "skip":
            continue
        target = decision.translation or unit.translated
        if not unit.source.strip() or not target:
            continue
        pairs[unit.source] = target
    return pairs


def _detect_patterns(
    pairs: dict[str, str], translations: Mapping[str, str]
) -> dict[str, PatternSchema]:
    groups: dict[_TemplateKey, dict[tuple[str, ...], PatternExampleSchema]] = (
        defaultdict(dict)
    )
    for source, target in pairs.items():
        words = source.split()
        if not 2 <= len(words) <= PATTERN_MAX_SOURCE_WORDS:
            continue
        example: PatternExampleSchema = {
            "source": source,
            "target": target,
        }
        for n_pre in range(len(words)):
            for n_suf in range(len(words) - n_pre):
                if n_pre + n_suf == 0:
                    continue
                if n_pre and words[0].startswith("<"):
                    continue
                if n_suf and words[len(words) - n_suf].startswith("<"):
                    continue
                key = (tuple(words[:n_pre]), tuple(words[len(words) - n_suf :]))
                slot = tuple(words[n_pre : len(words) - n_suf])
                groups[key].setdefault(slot, example)

    supported = {
        key: list(slots.values())
        for key, slots in groups.items()
        if len(slots) >= PATTERN_MIN_EXAMPLES
    }

    def specificity(key: _TemplateKey) -> tuple[int, int, _TemplateKey]:
        prefix_words, suffix_words = key
        return (len(prefix_words) + len(suffix_words), len(prefix_words), key)

    closed: dict[frozenset[str], _TemplateKey] = {}
    for key, examples in supported.items():
        signature = frozenset(e["source"] for e in examples)
        current = closed.get(signature)
        if current is None or specificity(key) > specificity(current):
            closed[signature] = key

    found: dict[str, PatternSchema] = {}
    for key in closed.values():
        examples = supported[key]
        targets = [e["target"] for e in examples]
        if len(set(targets)) < 2:
            continue
        tgt_pre = reduce(_common_prefix, targets)
        tgt_suf = reduce(_common_suffix, [t[len(tgt_pre) :] for t in targets])
        if not tgt_pre and not tgt_suf:
            continue
        prefix_words, suffix_words = key
        src_literals = (" ".join(prefix_words), " ".join(suffix_words))
        tgt_pre, examples = _complete_literal(
            src_literals[0], tgt_pre, tgt_suf, examples, translations, at_start=True
        )
        tgt_suf, examples = _complete_literal(
            src_literals[1], tgt_suf, tgt_pre, examples, translations, at_start=False
        )
        if not _tags_intact(src_literals, (tgt_pre, tgt_suf)):
            continue
        src_pattern = " ".join([*prefix_words, "{X}", *suffix_words])
        found[src_pattern] = PatternSchema(
            src_pattern=src_pattern,
            tgt_pattern=f"{tgt_pre}{{X}}{tgt_suf}",
            example_count=len(examples),
            examples=examples[:PATTERN_MAX_EXAMPLES],
        )
    return found


def _complete_literal(
    src_literal: str,
    literal: str,
    other: str,
    examples: list[PatternExampleSchema],
    translations: Mapping[str, str],
    *,
    at_start: bool,
) -> tuple[str, list[PatternExampleSchema]]:
    """Extend a literal that the character-level affix cut inside a word.

    A common prefix of 确认X and 确定X is 确, which teaches the translator a
    half word. When the source literal's own glossary translation extends
    the literal and most examples carry that full translation, the literal
    becomes the translation and only those examples back the template.
    Either signal alone misfires: glossary entries often differ from the
    compound form (Heavy -> 重型 vs 重X), and sibling terms share characters.
    """
    translation = translations.get(src_literal.casefold())
    if not literal or not translation or translation == literal:
        return literal, examples
    extends = translation.startswith if at_start else translation.endswith
    if not extends(literal):
        return literal, examples

    def carries(target: str) -> bool:
        has_affix = target.startswith if at_start else target.endswith
        return has_affix(translation) and len(target) > len(translation) + len(other)

    kept = [e for e in examples if carries(e["target"])]
    if len(kept) < PATTERN_MIN_EXAMPLES or len(kept) < PATTERN_LITERAL_MAJORITY * len(
        examples
    ):
        return literal, examples
    return translation, kept


def _tags_intact(src_literals: tuple[str, str], tgt_literals: tuple[str, str]) -> bool:
    """Reject templates whose `{X}` slot cuts through markup.

    Whitespace splitting can leave `<font color='#...'>Word` inside the slot
    while the character-level target affix stops mid-tag, yielding patterns
    such as `<font color='#{X}</font>` that teach the translator broken tags.
    """
    if any(
        part.count("<") != part.count(">") for part in (*src_literals, *tgt_literals)
    ):
        return False
    return validate_tags(" ".join(src_literals), " ".join(tgt_literals))[0]


def _common_prefix(a: str, b: str) -> str:
    return commonprefix([a, b])


def _common_suffix(a: str, b: str) -> str:
    if not a or not b:
        return ""
    i = 0
    for ca, cb in zip(reversed(a), reversed(b), strict=False):
        if ca != cb:
            break
        i += 1
    return a[len(a) - i :] if i > 0 else ""
