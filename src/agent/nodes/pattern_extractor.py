import json
from collections import defaultdict
from functools import reduce
from os.path import commonprefix
from typing import TypedDict

from loguru import logger
from pydantic import ValidationError

from src.agent._share import (
    PATTERN_CACHE_PATH,
    PATTERN_MAX_EXAMPLES,
    PATTERN_MAX_SOURCE_WORDS,
    PATTERN_MIN_EXAMPLES,
)
from src.models.agent import (
    NewAgentStateSchema,
    PatternExampleSchema,
    PatternSchema,
)

type _TemplateKey = tuple[tuple[str, ...], tuple[str, ...]]


class PatternExtractorOutputSchema(TypedDict):
    approved_pairs: dict[str, str]
    patterns: dict[str, PatternSchema]


def load_cached_patterns() -> dict[str, PatternSchema]:
    if not PATTERN_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(PATTERN_CACHE_PATH.read_text("utf-8"))
        patterns = [PatternSchema.model_validate(item) for item in raw]
    except (json.JSONDecodeError, OSError, ValidationError) as exc:
        logger.warning(f"Failed to load pattern cache: {exc}")
        return {}
    logger.info(f"Loaded {len(patterns)} cached patterns from {PATTERN_CACHE_PATH}")
    return {p.src_pattern: p for p in patterns}


def _save_cache(patterns: dict[str, PatternSchema]) -> None:
    try:
        PATTERN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = [p.model_dump() for p in patterns.values()]
        PATTERN_CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
        )
    except OSError as exc:
        logger.warning(f"Failed to save pattern cache: {exc}")


def pattern_extractor(state: NewAgentStateSchema) -> PatternExtractorOutputSchema:
    pairs = _collect_pairs(state)
    patterns = dict(state.patterns)

    changed = False
    if len(pairs) >= PATTERN_MIN_EXAMPLES:
        for src_pattern, mined in _detect_patterns(pairs).items():
            current = patterns.get(src_pattern)
            if current is not None and current.approved_count >= mined.approved_count:
                continue
            patterns[src_pattern] = mined
            changed = True
            if current is None:
                logger.info(
                    f'[PATTERN] "{mined.src_pattern}" → "{mined.tgt_pattern}"'
                    f" ({mined.approved_count} examples)"
                )
    if changed:
        _save_cache(patterns)

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


def _detect_patterns(pairs: dict[str, str]) -> dict[str, PatternSchema]:
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
        src_pattern = " ".join([*prefix_words, "{X}", *suffix_words])
        found[src_pattern] = PatternSchema(
            src_pattern=src_pattern,
            tgt_pattern=f"{tgt_pre}{{X}}{tgt_suf}",
            approved_count=len(examples),
            examples=examples[:PATTERN_MAX_EXAMPLES],
        )
    return found


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
