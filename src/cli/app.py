import csv
import io
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final

import typer
from loguru import logger

from src.agent.cli import app as agent_app
from src.core.aligner import BilingualAligner
from src.core.converter import CorpusConverter
from src.core.extractor import TermExtractor
from src.core.loc_writer import LocFileWriter
from src.core.parser import LocFileParser
from src.export.loader import load_corpus
from src.export.writer import CorpusWriter, GlossaryWriter
from src.models.corpus import BilingualCorpus
from src.models.glossary import Glossary

app = typer.Typer(
    name="x2loc", help="XCOM 2 localization file toolkit.", no_args_is_help=True
)

app.add_typer(agent_app, name="agent")

UPLOAD_CSV_COLUMNS: Final[list[str]] = [
    "context",
    "source",
    "target",
    "developer_comments",
]

logger.remove()
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <level>{level: <8}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    backtrace=True,
    enqueue=True,
    colorize=True,
)
logger.add(
    sink="logs/x2loc.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level: <8} {name}:{function}:{line} - {message}",
    level="INFO",
    backtrace=True,
    enqueue=True,
    rotation="10 MB",
    retention="1 days",
    encoding="utf-8",
)

# Base-game corpus JSON is written under this subdirectory so it never
# collides with a mod namespace. The leading underscore makes it visually
# distinct in directory listings (base game is special, not a mod).
BASE_GAME_OUTPUT_DIRNAME: Final[str] = "_base"


class OutputFormat(StrEnum):
    CSV = "csv"
    JSON = "json"


def _emit_output(text: str, output: Path | None, output_format: OutputFormat) -> None:
    """Write serialized command output to a file (or stdout when no path).

    CSV goes out with a UTF-8 BOM for Excel/Weblate compatibility, matching
    the writers in src/export.
    """
    if output:
        enc = "utf-8-sig" if output_format == OutputFormat.CSV else "utf-8"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding=enc)
        logger.info(f"Written to {output}")
    else:
        sys.stdout.write(text)


parser = LocFileParser()
aligner = BilingualAligner()
writer = CorpusWriter()
extractor = TermExtractor()
glossary_writer = GlossaryWriter()
converter = CorpusConverter()
loc_writer = LocFileWriter()


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

    _emit_output(text, output, output_format)


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

    _emit_output(text, output, output_format)


@app.command()
def extract(
    corpus_dirs: Annotated[
        list[Path],
        typer.Argument(
            help="Directories containing corpus JSON files (priority order)."
        ),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path.")
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.CSV,
    exclude_cosmetic: Annotated[
        bool,
        typer.Option("--exclude-cosmetic", help="Exclude cosmetic category terms."),
    ] = False,
) -> None:
    """Extract glossary terms from aligned corpus directories."""
    corpora: list[BilingualCorpus] = []

    for corpus_dir in corpus_dirs:
        if not corpus_dir.is_dir():
            logger.error(f"Corpus directory does not exist: {corpus_dir}")
            raise typer.Exit(1)

        json_files = sorted(corpus_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {corpus_dir}")
            continue

        for json_file in json_files:
            corpus = load_corpus(json_file)
            if corpus is not None:
                corpora.append(corpus)

    if not corpora:
        logger.error("No valid corpus files found")
        raise typer.Exit(1)

    glossary = extractor.extract(corpora)

    if exclude_cosmetic:
        glossary = Glossary(
            source_lang=glossary.source_lang,
            target_lang=glossary.target_lang,
            terms=[t for t in glossary.terms if t.category != "cosmetic"],
        )

    logger.info(f"Extracted {glossary.term_count} terms")

    if output_format == OutputFormat.CSV:
        text = glossary_writer.to_csv_string(glossary)
    else:
        text = glossary_writer.to_json_string(glossary)

    _emit_output(text, output, output_format)


if __name__ == "__main__":
    app()
