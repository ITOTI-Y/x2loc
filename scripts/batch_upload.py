"""Stage 2 batch upload: push every non-empty mod corpus to Weblate.

Reads the Stage 1 TSV (`output/batch_scan.tsv`) to pick namespaces
with entry_count > 0, then iterates `output/corpus/{namespace}/*.json`
and calls the same `_upload_single_corpus` helper that the regular
`x2loc upload` command uses. Skip-and-continue: a failure on mod N
is logged and the script moves on to mod N+1. A per-mod summary lands
in `output/batch_upload_report.tsv`.

Policy (matches user's Stage 2 decisions):
    - No per-mod glossary upload — Stage 3 will do a single consolidated
      glossary across all mods.
    - Mods with pre-existing .chn files ship their Chinese as initial
      translations via the normal upload path (method='replace' on
      Mode 1 first-time create; Mode 2 is non-destructive).
    - Base-game components (`base-xcom2-wotc`) stay untouched.

Re-running is safe: existing components fall through to Mode 2
(additive context diff → no-op if nothing new). Use this to retry
failed mods after fixing root causes.

Run:
    uv run python scripts/batch_upload.py
"""

from __future__ import annotations

import csv
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.cli.app import _upload_single_corpus  # noqa: E402
from src.models.weblate import WeblateConfigSchema  # noqa: E402
from src.services.weblate import WeblateAPIError, WeblateClient  # noqa: E402

SCAN_TSV: Path = ROOT / "output" / "batch_scan.tsv"
CORPUS_ROOT: Path = ROOT / "output" / "corpus"
REPORT_TSV: Path = ROOT / "output" / "batch_upload_report.tsv"
CONFIG_TOML: Path = ROOT / "configs" / "weblate.toml"

REPORT_COLUMNS: list[str] = [
    "mod_id",
    "namespace",
    "mod_title",
    "has_chn",
    "int_count",
    "entry_count",
    "components_uploaded",
    "status",
    "error",
]


def _load_config() -> WeblateConfigSchema:
    with CONFIG_TOML.open("rb") as f:
        data = tomllib.load(f)
    return WeblateConfigSchema(
        url=data["url"],
        token=data["token"],
        project_slug=data["project_slug"],
        license=data.get("license", ""),
        license_url=data.get("license_url", ""),
    )


def _load_scan_rows() -> list[dict[str, str]]:
    """Return Stage 1 TSV rows filtered to `status=ok` with entry_count>0."""
    if not SCAN_TSV.exists():
        raise FileNotFoundError(
            f"{SCAN_TSV} not found — run scripts/batch_scan.py first"
        )
    with SCAN_TSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [
            r
            for r in reader
            if r.get("status") == "ok" and int(r.get("entry_count") or 0) > 0
        ]
    return rows


def _upload_one_mod(
    client: WeblateClient,
    mod_row: dict[str, str],
    cfg: WeblateConfigSchema,
) -> dict[str, Any]:
    """Upload all corpus JSONs under this mod's namespace dir.

    Returns a report-row dict populated with the upload outcome.
    """
    report: dict[str, Any] = {
        "mod_id": mod_row["mod_id"],
        "namespace": mod_row["namespace"],
        "mod_title": mod_row["mod_title"],
        "has_chn": mod_row["has_chn"],
        "int_count": mod_row["int_count"],
        "entry_count": mod_row["entry_count"],
        "components_uploaded": 0,
        "status": "ok",
        "error": "",
    }

    namespace_dir = CORPUS_ROOT / mod_row["namespace"]
    if not namespace_dir.is_dir():
        report["status"] = "failed"
        report["error"] = f"corpus dir missing: {namespace_dir}"
        return report

    json_files = sorted(namespace_dir.glob("*.json"))
    if not json_files:
        report["status"] = "skipped"
        report["error"] = "no corpus json"
        return report

    uploaded = 0
    for json_file in json_files:
        try:
            _upload_single_corpus(
                client,
                json_file,
                target_lang="zh_Hans",
                method="replace",
                yes=True,
                license=cfg.license,
                license_url=cfg.license_url,
            )
            uploaded += 1
        except WeblateAPIError as e:
            report["status"] = "failed"
            report["error"] = f"weblate_api_error: {e}"
            report["components_uploaded"] = uploaded
            return report
        except Exception as e:
            report["status"] = "failed"
            report["error"] = f"{type(e).__name__}: {e}"
            report["components_uploaded"] = uploaded
            return report

    report["components_uploaded"] = uploaded
    return report


def main() -> None:
    # Tone down verbose per-file logging — we only want warnings and the
    # progress bar from this script's own prints.
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="WARNING",
        format="{time:HH:mm:ss} | {level: <7} | {message}",
    )

    cfg = _load_config()
    rows = _load_scan_rows()
    print(f"Stage 2 batch upload: {len(rows)} non-empty mods from {SCAN_TSV.name}")
    print(f"  Weblate URL: {cfg.url}")
    print(f"  Project: {cfg.project_slug}")
    print()

    report_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}

    with WeblateClient(cfg) as client:
        for i, mod_row in enumerate(rows, 1):
            try:
                report = _upload_one_mod(client, mod_row, cfg)
            except Exception:
                report = {
                    "mod_id": mod_row["mod_id"],
                    "namespace": mod_row["namespace"],
                    "mod_title": mod_row["mod_title"],
                    "has_chn": mod_row["has_chn"],
                    "int_count": mod_row["int_count"],
                    "entry_count": mod_row["entry_count"],
                    "components_uploaded": 0,
                    "status": "failed",
                    "error": f"uncaught_exception: {traceback.format_exc(limit=3)}",
                }
            report_rows.append(report)
            status: str = str(report["status"])
            status_counts[status] = status_counts.get(status, 0) + 1

            # Progress every 10 mods, plus the final tick.
            if i % 10 == 0 or i == len(rows):
                print(
                    f"  [{i}/{len(rows)}]  ok={status_counts['ok']} "
                    f"failed={status_counts['failed']} "
                    f"skipped={status_counts['skipped']}  "
                    f"last: {mod_row['namespace']}"
                )

    # Write report
    REPORT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(report_rows)

    total_components = sum(r["components_uploaded"] for r in report_rows)
    print()
    print(f"Report written to {REPORT_TSV}")
    print(
        f"  ok={status_counts['ok']}  failed={status_counts['failed']}  "
        f"skipped={status_counts['skipped']}  total_components={total_components}"
    )

    if status_counts["failed"]:
        print("\nFailed mods (first 20):")
        for r in report_rows:
            if r["status"] == "failed":
                print(f"  {r['namespace']:<45}  {r['error'][:100]}")
                if sum(1 for rr in report_rows if rr["status"] == "failed") > 20:
                    break


if __name__ == "__main__":
    main()
