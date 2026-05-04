"""Bootstrap data/patterns.json from existing approved Weblate translations.

Treats every translated unit in `temp/*.json` as an approved history
entry and detects prefix/suffix patterns by bucketing source words into
(prefix, suffix) tuples and applying a `next_pres` / `next_sufs`
diversity filter (see `_detect_patterns`). Output is written to
`data/patterns.json` so that fresh agent runs start with prior pattern
knowledge instead of an empty cache.

Note: the algorithm here is NOT identical to the agent's runtime
`pattern_extractor`, which uses pairwise comparison plus regex
fullmatch validation against history. Given the same input, the two
methods can produce different pattern sets — `data/patterns.json` may
diverge from what the online agent extracts during a session.

Run:
    uv run python scripts/prebuild_patterns.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.agent._share import PATTERN_CACHE_PATH, PATTERN_MIN_EXAMPLES  # noqa: E402
from src.agent.nodes._helpers import common_prefix as _common_prefix  # noqa: E402
from src.agent.nodes._helpers import common_suffix as _common_suffix  # noqa: E402

TEMP_DIR: Path = ROOT / "temp"


def _load_history() -> list[dict]:
    history: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for jf in sorted(TEMP_DIR.glob("*.json")):
        try:
            data = json.loads(jf.read_text("utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping {jf.name}: {exc}")
            continue
        kept = 0
        for entry in data:
            if not entry.get("translated"):
                continue
            src_list = entry.get("source") or []
            tgt_list = entry.get("target") or []
            if not src_list or not tgt_list:
                continue
            source = src_list[0]
            target = tgt_list[0]
            if not source or not target:
                continue
            key = (source, target)
            if key in seen:
                continue
            seen.add(key)
            history.append({"source": source, "target": target})
            kept += 1
        logger.info(f"{jf.name}: loaded {kept} approved pairs")
    return history


def _is_structurally_broken(src_pattern: str, tgt_pattern: str) -> bool:
    if any(c in src_pattern for c in ("<", ">", "=", '"')):
        return True
    if any(c in tgt_pattern for c in ("<", ">")):
        return True
    if src_pattern.count("(") != src_pattern.count(")"):
        return True
    if src_pattern.count("[") != src_pattern.count("]"):
        return True
    return src_pattern.startswith(("(", "[", "<"))


def _is_target_fragment(tgt_pre: str, tgt_suf: str) -> bool:
    for part in (tgt_pre, tgt_suf):
        if part and len(part) <= 1 and part.isascii() and part.isalpha():
            return True
    return False


def _detect_patterns(history: list[dict]) -> list[dict]:
    word_lists = [h["source"].split() for h in history]
    buckets: dict[tuple[tuple[str, ...], tuple[str, ...]], list[int]] = defaultdict(
        list
    )

    for idx, words in enumerate(word_lists):
        n = len(words)
        if n < 2:
            continue
        for p in range(n):
            for s in range(n - p):
                if p + s == 0:
                    continue
                if p + s >= n:
                    break
                pre = tuple(words[:p])
                suf = tuple(words[n - s :]) if s > 0 else ()
                buckets[(pre, suf)].append(idx)

    patterns: list[dict] = []
    seen_src: set[str] = set()

    for (pre, suf), idxs in buckets.items():
        if len(idxs) < PATTERN_MIN_EXAMPLES:
            continue

        next_pres = {word_lists[i][len(pre)] for i in idxs}
        next_sufs = {word_lists[i][len(word_lists[i]) - len(suf) - 1] for i in idxs}
        if len(next_pres) < 2 or len(next_sufs) < 2:
            continue

        targets = [history[i]["target"] for i in idxs]
        tgt_pre = targets[0]
        for t in targets[1:]:
            tgt_pre = _common_prefix(tgt_pre, t)
            if not tgt_pre:
                break
        tgt_suf = targets[0]
        for t in targets[1:]:
            tgt_suf = _common_suffix(tgt_suf, t)
            if not tgt_suf:
                break
        if not tgt_pre and not tgt_suf:
            continue
        if _is_target_fragment(tgt_pre, tgt_suf):
            continue

        src_pre = " ".join(pre) + " " if pre else ""
        src_suf = " " + " ".join(suf) if suf else ""
        src_pattern = f"{src_pre}{{X}}{src_suf}".strip()
        if src_pattern == "{X}" or src_pattern in seen_src:
            continue

        tgt_pattern = f"{tgt_pre}{{X}}{tgt_suf}"
        if _is_structurally_broken(src_pattern, tgt_pattern):
            continue
        seen_src.add(src_pattern)
        examples = [
            {"source": history[i]["source"], "target": history[i]["target"]}
            for i in idxs[:5]
        ]
        patterns.append(
            {
                "src_pattern": src_pattern,
                "tgt_pattern": tgt_pattern,
                "approved_count": len(idxs),
                "examples": examples,
            }
        )

    patterns.sort(key=lambda p: (-p["approved_count"], p["src_pattern"]))
    return patterns


def main() -> None:
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level: <7} | {message}",
    )

    if not TEMP_DIR.is_dir():
        raise FileNotFoundError(f"Missing temp dir: {TEMP_DIR}")

    history = _load_history()
    logger.info(f"Total approved pairs: {len(history)}")

    patterns = _detect_patterns(history)
    logger.info(
        f"Detected {len(patterns)} patterns (min_examples={PATTERN_MIN_EXAMPLES})"
    )

    out_path = ROOT / PATTERN_CACHE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patterns, ensure_ascii=False, indent=2), "utf-8")
    logger.info(f"Wrote {out_path}")

    if patterns:
        print("\nTop 10 patterns by approved_count:")
        for p in patterns[:10]:
            print(
                f'  [{p["approved_count"]:>4}] "{p["src_pattern"]}" → "{p["tgt_pattern"]}"'
            )


if __name__ == "__main__":
    main()
