from __future__ import annotations

import json
import re

from loguru import logger

from src.agent._share import PATTERN_CACHE_PATH, PATTERN_MIN_EXAMPLES
from src.agent.nodes._helpers import common_prefix, common_suffix
from src.models.agent import NewAgentStateSchema, PatternSchema


def load_cached_patterns() -> dict[str, PatternSchema]:
    if not PATTERN_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(PATTERN_CACHE_PATH.read_text("utf-8"))
        logger.info(f"Loaded {len(data)} cached patterns from {PATTERN_CACHE_PATH}")
        return {p["src_pattern"]: p for p in data}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to load pattern cache: {exc}")
        return {}


def _save_cache(patterns: list[PatternSchema]) -> None:
    try:
        PATTERN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PATTERN_CACHE_PATH.write_text(
            json.dumps(patterns, ensure_ascii=False, indent=2), "utf-8"
        )
    except OSError as exc:
        logger.warning(f"Failed to save pattern cache: {exc}")


def pattern_extractor(state: NewAgentStateSchema) -> NewAgentStateSchema:
    decisions = state.decisions
    if len(decisions) < PATTERN_MIN_EXAMPLES:
        return state

    new_patterns = _detect_patterns(decisions, state.patterns)
    if not new_patterns:
        return state

    state.patterns.update(new_patterns)
    return state


def _detect_patterns(
    history: list[dict],
    existing_patterns: list[PatternSchema],
) -> list[PatternSchema]:
    existing_src = {p["src_pattern"] for p in existing_patterns}
    found: list[PatternSchema] = []

    for i, a in enumerate(history):
        for b in history[i + 1 :]:
            words_a = a["source"].split()
            words_b = b["source"].split()

            prefix: list[str] = []
            for wa, wb in zip(words_a, words_b, strict=False):
                if wa != wb:
                    break
                prefix.append(wa)

            suffix: list[str] = []
            for wa, wb in zip(reversed(words_a), reversed(words_b), strict=False):
                if wa != wb:
                    break
                suffix.append(wa)
            suffix.reverse()

            if len(prefix) + len(suffix) == 0:
                continue
            if len(prefix) + len(suffix) >= min(len(words_a), len(words_b)):
                continue

            src_prefix = " ".join(prefix) + " " if prefix else ""
            src_suffix = " " + " ".join(suffix) if suffix else ""
            src_pattern = f"{src_prefix}{{X}}{src_suffix}".strip()

            if src_pattern in existing_src or src_pattern == "{X}":
                continue

            regex_parts = []
            if prefix:
                regex_parts.append(re.escape(" ".join(prefix)))
            regex_parts.append("(.+)")
            if suffix:
                regex_parts.append(re.escape(" ".join(suffix)))
            pattern_re = re.compile(r"^" + r"\s+".join(regex_parts) + r"$")

            matches = [h for h in history if pattern_re.fullmatch(h["source"])]
            if len(matches) < PATTERN_MIN_EXAMPLES:
                continue

            tgt_pre = common_prefix(a["target"], b["target"])
            tgt_suf = common_suffix(a["target"], b["target"])
            if not tgt_pre and not tgt_suf:
                continue

            tgt_pattern = f"{tgt_pre}{{X}}{tgt_suf}"
            existing_src.add(src_pattern)
            found.append(
                {
                    "src_pattern": src_pattern,
                    "tgt_pattern": tgt_pattern,
                    "approved_count": len(matches),
                    "examples": [
                        {"source": m["source"], "target": m["target"]}
                        for m in matches[:5]
                    ],
                }
            )
            logger.info(
                f'[PATTERN] "{src_pattern}" → "{tgt_pattern}" ({len(matches)} examples)'
            )

    return found
