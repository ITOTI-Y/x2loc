from __future__ import annotations

import json
import re

from loguru import logger

from src.agent._share import PATTERN_CACHE_PATH, PATTERN_MIN_EXAMPLES
from src.agent.nodes._helpers import common_prefix, common_suffix
from src.agent.state import AgentState, SessionPattern


def load_cached_patterns() -> list[SessionPattern]:
    if not PATTERN_CACHE_PATH.exists():
        return []
    try:
        data = json.loads(PATTERN_CACHE_PATH.read_text("utf-8"))
        logger.info(f"Loaded {len(data)} cached patterns from {PATTERN_CACHE_PATH}")
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to load pattern cache: {exc}")
        return []


def _save_cache(patterns: list[SessionPattern]) -> None:
    try:
        PATTERN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PATTERN_CACHE_PATH.write_text(
            json.dumps(patterns, ensure_ascii=False, indent=2), "utf-8"
        )
    except OSError as exc:
        logger.warning(f"Failed to save pattern cache: {exc}")


def pattern_extractor(state: AgentState) -> dict:
    history = state.get("approved_history", [])
    if len(history) < PATTERN_MIN_EXAMPLES:
        return {}

    new_patterns = _detect_patterns(history, state.get("session_patterns", []))
    if not new_patterns:
        return {}

    merged = list(state.get("session_patterns", [])) + new_patterns
    _save_cache(merged)
    return {"session_patterns": merged}


def _detect_patterns(
    history: list[dict],
    existing_patterns: list[SessionPattern],
) -> list[SessionPattern]:
    existing_src = {p["src_pattern"] for p in existing_patterns}
    found: list[SessionPattern] = []

    for i, a in enumerate(history):
        for b in history[i + 1 :]:
            words_a = a["source"].split()
            words_b = b["source"].split()

            prefix: list[str] = []
            for wa, wb in zip(words_a, words_b, strict=True):
                if wa != wb:
                    break
                prefix.append(wa)

            suffix: list[str] = []
            for wa, wb in zip(reversed(words_a), reversed(words_b), strict=True):
                if wa != wb:
                    break
                suffix.insert(0, wa)

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
            regex = r"^" + r"\s+".join(regex_parts) + r"$"

            matches = [h for h in history if re.fullmatch(regex, h["source"])]
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
