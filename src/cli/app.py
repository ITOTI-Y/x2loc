import json
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from src._share import EXT_LANG_MAP, LANG_EXT_MAP
from src.core.aligner import BilingualAligner
from src.core.parser import LocFileParser
from src.export.writer import CorpusWriter

app = typer.Typer(
    name="x2loc", help="XCOM 2 localization file toolkit.", no_args_is_help=True
)


class OutputFormat(StrEnum):
    CSV = "csv"
    JSON = "json"


parser = LocFileParser()
aligner = BilingualAligner()
writer = CorpusWriter()


@app.command()
def parse(
    path: Annotated[Path, typer.Argument(help="Localization file to parse.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path.")
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format.")
    ] = OutputFormat.JSON,
) -> None:
    """Parse a single localization file."""
    loc_file = parser.parse(path)

    if output_format == OutputFormat.JSON:
        text = json.dumps(
            loc_file.model_dump(mode="json"),
            indent=4,
            ensure_ascii=False,
        )
    else:
        import csv
        import io

        buf = io.StringIO()
        columns = [
            "section",
            "key",
            "value",
            "is_array",
            "is_append",
            "line_number",
            "has_placeholders",
        ]
        w = csv.DictWriter(buf, fieldnames=columns)
        w.writeheader()
        for section in loc_file.sections:
            for entry in section.entries:
                w.writerow(
                    {
                        "section": section.header.raw,
                        "key": entry.key,
                        "value": entry.value,
                        "is_array": entry.is_array,
                        "is_append": entry.is_append,
                        "line_number": entry.line_number,
                        "has_placeholders": bool(entry.placeholders),
                    }
                )
        text = buf.getvalue()

    if output:
        enc = "utf-8-sig" if output_format == OutputFormat.CSV else "utf-8"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding=enc)
        logger.info(f"Written to {output}")
    else:
        sys.stdout.write(text)


@app.command()
def align(
    source: Annotated[Path, typer.Argument(help="Source language file.")],
    target: Annotated[Path, typer.Argument(help="Target language file.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path.")
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.JSON,
) -> None:
    """Align two localization files."""
    src_file = parser.parse(source)
    tgt_file = parser.parse(target)
    corpus = aligner.align(src_file, tgt_file)

    if output_format == OutputFormat.CSV:
        text = writer.to_csv_string(corpus)
    else:
        text = writer.to_json_string(corpus)

    if output:
        if output_format == OutputFormat.CSV:
            writer.write_csv(corpus, output)
        else:
            writer.write_json(corpus, output)
        logger.info(f"Written to {output}")
    else:
        sys.stdout.write(text)


@app.command("align-dir")
def align_dir(
    source_dir: Annotated[Path, typer.Argument(help="Source language directory.")],
    target_dir: Annotated[Path, typer.Argument(help="Target language directory.")],
    target_lang: Annotated[
        str | None,
        typer.Option(
            "--target-lang",
            "-t",
            help="BCP-47 target language code (e.g. zh_Hans, ja).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Output directory."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.JSON,
) -> None:
    """Batch-align all files in source and target directories."""
    if not source_dir.is_dir():
        logger.error(f"Source directory does not exist: {source_dir}")
        raise typer.Exit(1)

    target_ext = _resolve_target_ext(target_lang, target_dir)
    effective_target_lang = target_lang or LANG_EXT_MAP.get(target_ext, "")
    if not effective_target_lang:
        logger.error(f"Cannot determine target language from extension: {target_ext}")
        raise typer.Exit(1)

    out = output_dir or Path("output")
    out.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        f for f in source_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    )

    matched = 0
    source_only_count = 0
    skipped = 0

    for src_path in source_files:
        src_ext = src_path.suffix.lstrip(".").lower()
        if src_ext not in LANG_EXT_MAP:
            logger.debug(f"Skipping non-localization file: {src_path.name}")
            skipped += 1
            continue

        tgt_path = target_dir / f"{src_path.stem}.{target_ext}"
        src_file = parser.parse(src_path)

        if tgt_path.exists():
            tgt_file = parser.parse(tgt_path)
            corpus = aligner.align(src_file, tgt_file)
            matched += 1
        else:
            corpus = aligner.align(src_file, target_lang=effective_target_lang)
            source_only_count += 1
            logger.warning(
                f"Target not found: {tgt_path.name}, producing source-only corpus"
            )

        out_name = f"{src_path.stem}.{output_format.value}"
        out_path = out / out_name
        if output_format == OutputFormat.CSV:
            writer.write_csv(corpus, out_path)
        else:
            writer.write_json(corpus, out_path)

    logger.info(
        f"Done: matched={matched}, source_only={source_only_count}, skipped={skipped}"
    )


def _resolve_target_ext(target_lang: str | None, target_dir: Path) -> str:
    """Determine target file extension from language or directory contents.

    Priority:
        1. --target-lang flag -> EXT_LANG_MAP lookup
        2. Infer from existing files in target_dir (majority vote)
        3. Error if cannot determine
    """
    if target_lang:
        ext = EXT_LANG_MAP.get(target_lang)
        if not ext:
            logger.error(f"Unknown target language: {target_lang}")
            raise typer.Exit(1)
        return ext

    if not target_dir.is_dir():
        logger.error(
            f"Target directory does not exist and --target-lang not provided: "
            f"{target_dir}"
        )
        raise typer.Exit(1)

    exts = [
        f.suffix.lstrip(".").lower()
        for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lstrip(".").lower() in LANG_EXT_MAP
    ]

    if not exts:
        logger.error(
            "Target directory is empty or has no localization files, "
            "and --target-lang not provided"
        )
        raise typer.Exit(1)

    return Counter(exts).most_common(1)[0][0]
