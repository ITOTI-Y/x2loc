"""Stage 1 dry-run: batch-align every mod under data/RealModFiles/.

One-shot script — not part of the CLI surface. Walks `data/RealModFiles/*/`,
runs mod_resolver + aligner for each, writes corpus JSON under
`output/corpus/{namespace}/` and a summary TSV to
`output/batch_scan.tsv`. Skips mods without a Localization/ dir or
whose .XComMod manifest can't be parsed, recording the skip reason in
the TSV rather than failing the whole run.

Run:
    uv run python scripts/batch_scan.py

TSV columns (tab-separated):
    mod_id              directory name under data/RealModFiles/
    namespace           resolved mod namespace ('' on failure)
    mod_title           .XComMod Title field ('' on failure)
    loc_subdir          Localization subpath relative to mod root
    has_chn             y/n — does target_dir contain any .chn files?
    int_count           number of .int source files processed
    entry_count         BilingualCorpus entries across all files
    source_only_count   entries without a target match
    status              ok | skipped
    reason              empty on success, else the skip/error reason
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.core.aligner import BilingualAligner  # noqa: E402
from src.core.mod_resolver import ModResolveError, resolve_mod  # noqa: E402
from src.core.parser import LocFileParser  # noqa: E402
from src.export.writer import CorpusWriter  # noqa: E402
REAL_MOD_ROOT: Path = ROOT / "data" / "RealModFiles"
OUTPUT_CORPUS: Path = ROOT / "output" / "corpus"
OUTPUT_TSV: Path = ROOT / "output" / "batch_scan.tsv"

TSV_COLUMNS: list[str] = [
    "mod_id",
    "namespace",
    "mod_title",
    "loc_subdir",
    "has_chn",
    "int_count",
    "entry_count",
    "source_only_count",
    "status",
    "reason",
]


def _find_localization_dir(mod_root: Path) -> Path | None:
    """Return the directory under `mod_root` that contains .int files.

    Covers the two common layouts:
      - `mod_root/Localization/*.int`                 (most mods)
      - `mod_root/Localization/{SubMod}/*.int`        (LWOTC-style nesting)

    Picks the first subdir with .int files, walking up to 3 levels
    under `Localization/`. Returns None when no .int file is found.
    """
    loc_root = mod_root / "Localization"
    if not loc_root.is_dir():
        # Some mods use a different capitalization or no dir at all.
        for child in mod_root.iterdir():
            if child.is_dir() and child.name.lower() == "localization":
                loc_root = child
                break
        else:
            return None

    # Direct: Localization/*.int
    direct_ints = [
        f for f in loc_root.iterdir() if f.is_file() and f.suffix.lower() == ".int"
    ]
    if direct_ints:
        return loc_root

    # Nested: Localization/*/*.int  (LWOTC etc.)
    for child in sorted(loc_root.iterdir()):
        if not child.is_dir():
            continue
        nested_ints = [
            f for f in child.iterdir() if f.is_file() and f.suffix.lower() == ".int"
        ]
        if nested_ints:
            return child

    return None


def _has_chn_files(loc_dir: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() == ".chn" for f in loc_dir.iterdir())


def _process_mod(
    mod_dir: Path,
    parser: LocFileParser,
    aligner: BilingualAligner,
    writer: CorpusWriter,
) -> dict[str, str]:
    """Align one mod, write its corpus, return a TSV row dict."""
    row: dict[str, str] = dict.fromkeys(TSV_COLUMNS, "")
    row["mod_id"] = mod_dir.name

    loc_dir = _find_localization_dir(mod_dir)
    if loc_dir is None:
        row["status"] = "skipped"
        row["reason"] = "no_localization_dir"
        return row

    row["loc_subdir"] = loc_dir.relative_to(mod_dir).as_posix()
    row["has_chn"] = "y" if _has_chn_files(loc_dir) else "n"

    try:
        mod_info = resolve_mod(loc_dir, mod_dir)
    except ModResolveError as e:
        row["status"] = "skipped"
        row["reason"] = f"mod_resolve_error: {e}"
        return row

    row["namespace"] = mod_info.namespace
    row["mod_title"] = mod_info.mod_title

    int_files = sorted(
        f for f in loc_dir.iterdir() if f.is_file() and f.suffix.lower() == ".int"
    )
    if not int_files:
        row["status"] = "skipped"
        row["reason"] = "no_int_files"
        return row

    row["int_count"] = str(len(int_files))

    out_dir = OUTPUT_CORPUS / mod_info.namespace
    out_dir.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    total_source_only = 0

    for int_path in int_files:
        chn_path = loc_dir / f"{int_path.stem}.chn"
        try:
            src_file = parser.parse(int_path)
        except Exception as e:
            logger.warning(f"{mod_dir.name}/{int_path.name}: parse failed ({e})")
            continue

        if chn_path.exists():
            try:
                tgt_file = parser.parse(chn_path)
                corpus = aligner.align(src_file, tgt_file, mod_info=mod_info)
            except Exception as e:
                logger.warning(
                    f"{mod_dir.name}/{chn_path.name}: parse failed ({e}); falling back to source-only"
                )
                corpus = aligner.align(
                    src_file, target_lang="zh_Hans", mod_info=mod_info
                )
        else:
            corpus = aligner.align(src_file, target_lang="zh_Hans", mod_info=mod_info)

        total_entries += len(corpus.entries)
        total_source_only += len(corpus.source_only)

        writer.write_json(corpus, out_dir / f"{int_path.stem}.json")

    row["entry_count"] = str(total_entries)
    row["source_only_count"] = str(total_source_only)
    row["status"] = "ok"
    return row


def main() -> None:
    OUTPUT_CORPUS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)

    # Silence per-file parser warnings at INFO; keep loguru WARN+ visible.
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="WARNING",
        format="{time:HH:mm:ss} | {level: <7} | {message}",
    )

    parser = LocFileParser()
    aligner = BilingualAligner()
    writer = CorpusWriter()

    mod_dirs = sorted(d for d in REAL_MOD_ROOT.iterdir() if d.is_dir())
    print(f"Found {len(mod_dirs)} mod directories under {REAL_MOD_ROOT}")

    rows: list[dict[str, str]] = []
    status_counts = {"ok": 0, "skipped": 0}
    for i, mod_dir in enumerate(mod_dirs, 1):
        row = _process_mod(mod_dir, parser, aligner, writer)
        rows.append(row)
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if i % 50 == 0 or i == len(mod_dirs):
            print(
                f"  [{i}/{len(mod_dirs)}]  ok={status_counts['ok']} "
                f"skipped={status_counts['skipped']}"
            )

    with OUTPUT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"\nSummary written to {OUTPUT_TSV}")
    print(f"  ok={status_counts['ok']}  skipped={status_counts['skipped']}")

    # Breakdown of skip reasons
    reason_counts: dict[str, int] = {}
    for row in rows:
        if row["status"] == "skipped":
            reason = row["reason"].split(":")[0]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        print("  Skip reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason}: {count}")

    # Aggregate stats for ok mods
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        total_int = sum(int(r["int_count"]) for r in ok_rows)
        total_entries = sum(int(r["entry_count"]) for r in ok_rows)
        with_chn = sum(1 for r in ok_rows if r["has_chn"] == "y")
        print(
            f"\nOK mods: {len(ok_rows)} total  |  "
            f"{total_int} .int files  |  "
            f"{total_entries} entries  |  "
            f"{with_chn} have pre-existing .chn"
        )

    # Namespace uniqueness check
    namespaces = [r["namespace"] for r in rows if r["namespace"]]
    if len(namespaces) != len(set(namespaces)):
        from collections import Counter

        dup = [(n, c) for n, c in Counter(namespaces).items() if c > 1]
        print(f"\n⚠️  Duplicate namespaces detected: {dup}")


if __name__ == "__main__":
    main()
