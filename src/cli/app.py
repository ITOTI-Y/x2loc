import csv
import io
import json
import sys
import tomllib
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from loguru import logger

from src._share import EXT_LANG_MAP, LANG_EXT_MAP
from src.core._share import iter_compound_keys
from src.core.aligner import BilingualAligner
from src.core.converter import TRANSLATABLE_STRUCT_FIELDS, CorpusConverter
from src.core.extractor import TermExtractor
from src.core.loc_writer import LocFileWriter
from src.core.mod_resolver import ModResolveError, resolve_mod
from src.core.parser import LocFileParser
from src.export.writer import CorpusWriter, GlossaryWriter
from src.models.corpus import BilingualCorpus
from src.models.file import LocalizationFile
from src.models.glossary import Glossary
from src.models.mod import ModInfoSchema
from src.models.weblate import WeblateConfigSchema
from src.services.weblate import WeblateAPIError, WeblateClient

app = typer.Typer(
    name="x2loc", help="XCOM 2 localization file toolkit.", no_args_is_help=True
)

UPLOAD_CSV_COLUMNS: Final[list[str]] = [
    "context",
    "source",
    "target",
    "developer_comments",
]

# Base-game corpus JSON is written under this subdirectory so it never
# collides with a mod namespace. The leading underscore makes it visually
# distinct in directory listings (base game is special, not a mod).
BASE_GAME_OUTPUT_DIRNAME: Final[str] = "_base"


class OutputFormat(StrEnum):
    CSV = "csv"
    JSON = "json"


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
    sandbox_root: Annotated[
        Path | None,
        typer.Option(
            "--sandbox-root",
            help=(
                "Absolute root of the current job's sandbox — walk-up when "
                "resolving the .XComMod manifest will not cross this path. "
                "Required for mod uploads; for base game use --base-game "
                "instead."
            ),
        ),
    ] = None,
    steam_id: Annotated[
        str | None,
        typer.Option(
            "--steam-id",
            help=(
                "Explicit Steam Workshop ID. Overrides filesystem-based "
                "detection. Use when the mod was extracted from a zip whose "
                "directory name isn't the Steam ID and the .XComMod carries "
                "`publishedFileId=0`."
            ),
        ),
    ] = None,
    base_game: Annotated[
        bool,
        typer.Option(
            "--base-game",
            help=(
                "Treat source_dir as the official XCOM 2 base game files "
                "(no .XComMod, fixed namespace). Output is written to "
                "output/corpus/_base/ regardless of --output-dir subpath."
            ),
        ),
    ] = False,
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
        typer.Option("--output-dir", "-o", help="Output directory root."),
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

    mod_info = _resolve_mod_info_for_align(
        source_dir=source_dir,
        sandbox_root=sandbox_root,
        steam_id_override=steam_id,
        base_game=base_game,
    )

    target_ext = _resolve_target_ext(target_lang, target_dir)
    effective_target_lang = target_lang or LANG_EXT_MAP.get(target_ext, "")
    if not effective_target_lang:
        logger.error(f"Cannot determine target language from extension: {target_ext}")
        raise typer.Exit(1)

    # Namespaced output: output/corpus/{namespace}/*.json (or _base/ for base
    # game). Preserves the mod identity alongside its corpus data so
    # downstream upload/download know which Weblate components to target.
    namespace_dirname = BASE_GAME_OUTPUT_DIRNAME if base_game else mod_info.namespace
    out_root = output_dir or Path("output") / "corpus"
    out = out_root / namespace_dirname
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
            corpus = aligner.align(src_file, tgt_file, mod_info=mod_info)
            matched += 1
        else:
            corpus = aligner.align(
                src_file,
                target_lang=effective_target_lang,
                mod_info=mod_info,
            )
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
        f"Done: namespace={mod_info.namespace}, "
        f"matched={matched}, source_only={source_only_count}, skipped={skipped}"
    )


def _resolve_mod_info_for_align(
    source_dir: Path,
    sandbox_root: Path | None,
    steam_id_override: str | None,
    base_game: bool,
) -> ModInfoSchema:
    """Resolve mod identity for `align-dir`, honoring --base-game.

    Base game has no manifest and is always tagged with the fixed
    `base-xcom2-wotc` namespace; --steam-id and --sandbox-root are
    ignored in that branch. Everything else runs through the sandbox-
    bounded walk-up resolver.
    """
    if base_game:
        if steam_id_override:
            logger.warning("--steam-id is ignored when --base-game is set")
        return ModInfoSchema.base_game()

    if sandbox_root is None:
        logger.error(
            "--sandbox-root is required (or pass --base-game for the "
            "official XCOM 2 game files)"
        )
        raise typer.Exit(1)

    try:
        return resolve_mod(
            source_dir, sandbox_root, steam_id_override=steam_id_override
        )
    except ModResolveError as e:
        logger.error(f"Failed to resolve mod identity: {e}")
        raise typer.Exit(1) from e


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
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                corpus = BilingualCorpus.model_validate(data)
                corpora.append(corpus)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

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

    if output:
        if output_format == OutputFormat.CSV:
            glossary_writer.write_csv(glossary, output)
        else:
            glossary_writer.write_json(glossary, output)
        logger.info(f"Written to {output}")
    else:
        sys.stdout.write(text)


@app.command()
def upload(
    corpus_dir: Annotated[
        Path, typer.Argument(help="Directory containing corpus JSON files.")
    ],
    target_lang: Annotated[
        str, typer.Option("--target-lang", "-t", help="BCP-47 target language.")
    ],
    glossary: Annotated[
        Path | None,
        typer.Option("--glossary", help="Glossary CSV produced by `extract`."),
    ] = None,
    url: Annotated[str | None, typer.Option("--url")] = None,
    token: Annotated[str | None, typer.Option("--token")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="TOML config file.")
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help=(
                "Upload method: replace|translate|suggest|fuzzy. Default is "
                "`replace` — it is the only method Weblate reliably applies "
                "for bilingual CSV first-time uploads. `translate` may appear "
                "as a no-op on fresh components."
            ),
        ),
    ] = "replace",
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip confirmation prompts.")
    ] = False,
) -> None:
    """Upload aligned corpora (+ optional glossary) to Weblate."""
    if not corpus_dir.is_dir():
        logger.error(f"Corpus directory does not exist: {corpus_dir}")
        raise typer.Exit(1)

    cfg = _load_weblate_config(url, token, project, config)

    with WeblateClient(cfg) as client:
        try:
            _ensure_project(client, cfg)

            json_files = sorted(corpus_dir.glob("*.json"))
            if not json_files:
                logger.error(f"No corpus JSON files in {corpus_dir}")
                raise typer.Exit(1)

            # The glossary is uploaded as a single per-mod component
            # (`glossary-{namespace}`), so we need the namespace up front.
            # It is read from the first corpus JSON — align-dir writes all
            # sibling corpora under one namespace dir, so they must all
            # agree. A mismatch is treated as a data error.
            namespace = _read_namespace_from_corpora(json_files)

            for json_file in json_files:
                _upload_single_corpus(
                    client,
                    json_file,
                    target_lang,
                    method,
                    yes,
                    license=cfg.license,
                    license_url=cfg.license_url,
                )

            if glossary is not None:
                _upload_glossary(
                    client,
                    glossary,
                    namespace,
                    target_lang,
                    method,
                    license=cfg.license,
                    license_url=cfg.license_url,
                )

        except WeblateAPIError as e:
            logger.error(f"Weblate API error: {e}")
            raise typer.Exit(1) from e


def _read_namespace_from_corpora(json_files: list[Path]) -> str:
    """Extract the namespace from a batch of corpus JSON files.

    All files in the same corpus dir must share one namespace — they
    were written there by align-dir, which always uses
    `output/corpus/{namespace}/`. A mismatch indicates the directory
    was hand-edited or merged incorrectly; we fail loud.

    Parsing goes through the schema so legacy JSON produced before the
    namespace field existed still validates and picks up the base-game
    default.
    """
    seen: set[str] = set()
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            corpus = BilingualCorpus.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to read namespace from {jf.name}: {e}")
            continue
        seen.add(corpus.namespace)
    if not seen:
        logger.error(
            "No namespace found in any corpus JSON; rerun align-dir with "
            "--sandbox-root or --base-game."
        )
        raise typer.Exit(1)
    if len(seen) > 1:
        logger.error(
            f"Corpus dir contains multiple namespaces {sorted(seen)}; each "
            "namespace must live in its own directory."
        )
        raise typer.Exit(1)
    return next(iter(seen))


@app.command()
def download(
    target_lang: Annotated[
        str, typer.Option("--target-lang", "-t", help="BCP-47 target language.")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="CSV output directory.")
    ],
    url: Annotated[str | None, typer.Option("--url")] = None,
    token: Annotated[str | None, typer.Option("--token")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    component: Annotated[
        str | None,
        typer.Option("--component", help="Only download this component slug."),
    ] = None,
) -> None:
    """Download translations from Weblate as CSV files."""
    cfg = _load_weblate_config(url, token, project, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    with WeblateClient(cfg) as client:
        try:
            components = client.list_components()
        except WeblateAPIError as e:
            logger.error(f"Weblate API error: {e}")
            raise typer.Exit(1) from e

        total = 0
        for comp in components:
            slug = comp["slug"]
            if component is not None and slug != component:
                continue
            if comp.get("is_glossary"):
                continue
            try:
                data = client.download_file(slug, target_lang)
            except WeblateAPIError as e:
                logger.warning(f"download {slug}: {e}")
                continue
            out_path = output_dir / f"{slug}.csv"
            out_path.write_bytes(data)
            logger.info(f"Downloaded {out_path.name} ({len(data)} bytes)")
            total += 1

        logger.info(f"Downloaded {total} component(s) to {output_dir}")


@app.command()
def writeback(
    source_dir: Annotated[Path, typer.Argument(help="Source .int file directory.")],
    translations_dir: Annotated[
        Path, typer.Argument(help="Translations CSV directory (from `download`).")
    ],
    target_lang: Annotated[
        str, typer.Option("--target-lang", "-t", help="BCP-47 target language.")
    ],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Output directory.")
    ] = None,
) -> None:
    """Apply translated CSVs back onto source files and write target loc files."""
    if not source_dir.is_dir():
        logger.error(f"Source directory does not exist: {source_dir}")
        raise typer.Exit(1)
    if not translations_dir.is_dir():
        logger.error(f"Translations directory does not exist: {translations_dir}")
        raise typer.Exit(1)

    target_ext = EXT_LANG_MAP.get(target_lang)
    if not target_ext:
        logger.error(f"Unknown target language: {target_lang}")
        raise typer.Exit(1)

    out_dir = output_dir or Path("output") / "writeback"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        f
        for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lstrip(".").lower() in LANG_EXT_MAP
    )

    written = 0
    skipped = 0
    total_units = 0
    translated_units = 0

    for src_path in source_files:
        csv_path = translations_dir / f"{src_path.stem}.csv"
        if not csv_path.exists():
            logger.warning(f"No CSV for {src_path.name}, skipping")
            skipped += 1
            continue

        source_file = parser.parse(src_path)
        translations = _parse_translation_csv(csv_path)

        target_path = out_dir / f"{src_path.stem}.{target_ext}"
        target_file = converter.build_target_file(
            source=source_file,
            translations=translations,
            target_lang=target_lang,
            target_path=target_path,
        )
        loc_writer.write(target_file, target_path)

        file_total = source_file.entry_count
        file_done = _count_translated(source_file, translations)
        total_units += file_total
        translated_units += file_done
        ratio = (file_done / file_total * 100) if file_total else 0.0
        logger.info(
            f"Wrote {target_path.name} "
            f"({file_done}/{file_total} translated, {ratio:.1f}%)"
        )
        written += 1

    overall = (translated_units / total_units * 100) if total_units else 0.0
    logger.info(
        f"Total: {translated_units}/{total_units} translated "
        f"({overall:.1f}%), {written} written, {skipped} skipped"
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


def _load_weblate_config(
    url: str | None,
    token: str | None,
    project: str | None,
    config_path: Path | None,
) -> WeblateConfigSchema:
    """Resolve Weblate config from CLI flags or TOML file.

    Priority: explicit flags > config file > error. TOML-only fields
    (`license`, `license_url`) are read from the file when present.
    """
    if config_path is not None:
        if not config_path.is_file():
            logger.error(f"Config file not found: {config_path}")
            raise typer.Exit(1)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return WeblateConfigSchema(
            url=url or data["url"],
            token=token or data["token"],
            project_slug=project or data["project_slug"],
            license=data.get("license", ""),
            license_url=data.get("license_url", ""),
        )
    if url and token and project:
        return WeblateConfigSchema(url=url, token=token, project_slug=project)

    logger.error("Weblate connection requires --url/--token/--project or --config")
    raise typer.Exit(1)


def _make_csv_writer(buf: io.StringIO) -> csv.DictWriter:
    """Build a DictWriter that quotes every field.

    Weblate's `csv` file format auto-detects the delimiter by character
    frequency on the first data row. If a row contains more spaces than
    commas (very common for short glossary terms like `(VIP Capture Only)`),
    Weblate picks SPACE as the delimiter and everything downstream is
    garbled. Forcing `QUOTE_ALL` eliminates the ambiguity — every field is
    wrapped in `"..."`, so the sniffer unambiguously sees comma.
    """
    return csv.DictWriter(buf, fieldnames=UPLOAD_CSV_COLUMNS, quoting=csv.QUOTE_ALL)


def _units_to_source_csv_bytes(
    units: list[tuple[str, str, str, str]],
) -> bytes:
    """CSV for initial component creation (source-language docfile).

    Weblate's `csv` file format treats the CSV as a **language-specific
    file**: the `target` column holds the actual text for that file's
    language. When creating a component with source_language=en, the
    docfile IS the English-language file, so the `target` column must
    contain the English strings. We also write `source` = English for
    consistency/reference.
    """
    buf = io.StringIO()
    writer = _make_csv_writer(buf)
    writer.writeheader()
    for context, source, _target, note in units:
        if not source:
            continue
        writer.writerow(
            {
                "context": context,
                "source": source,
                "target": source,  # Weblate reads target col as file content
                "developer_comments": note,
            }
        )
    return buf.getvalue().encode("utf-8")


def _units_to_translation_csv_bytes(
    units: list[tuple[str, str, str, str]],
) -> bytes:
    """CSV for target-language translation upload.

    The target column holds the target-language text (e.g. Chinese); the
    source column holds the English reference for matching. Rows with an
    empty target are skipped because they carry no translation.
    """
    buf = io.StringIO()
    writer = _make_csv_writer(buf)
    writer.writeheader()
    for context, source, target, note in units:
        if not target:
            continue
        writer.writerow(
            {
                "context": context,
                "source": source,
                "target": target,
                "developer_comments": note,
            }
        )
    return buf.getvalue().encode("utf-8")


def _parse_translation_csv(path: Path) -> dict[str, str]:
    """Parse a downloaded Weblate CSV into {context: target} (skip empty targets)."""
    translations: dict[str, str] = {}
    # Weblate emits UTF-8 without BOM; utf-8-sig is forgiving either way.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            context = row.get("context") or ""
            target = row.get("target") or ""
            if not context or not target:
                continue
            if context in translations:
                logger.warning(f"Duplicate context {context}, last wins")
            translations[context] = target
    return translations


def _ensure_project(client: WeblateClient, cfg: WeblateConfigSchema) -> None:
    project = client.get_project()
    if project is None:
        logger.info(f"Creating Weblate project '{cfg.project_slug}'")
        client.create_project(name=cfg.project_slug, slug=cfg.project_slug)


def _upload_single_corpus(
    client: WeblateClient,
    corpus_path: Path,
    target_lang: str,
    method: str,
    yes: bool,
    license: str = "",
    license_url: str = "",
) -> None:
    """Upload one corpus JSON file as a Weblate component.

    Behavior:
        - **Mode 1 (fresh)**: when the component does not exist, create it
          via `create_component` with the full source CSV (fast bulk load).
        - **Mode 2 (incremental)**: when the component already exists,
          diff local units against existing Weblate contexts and POST
          `create_unit` for the delta only. Existing units' target
          translations are updated via `upload_file(method=translate)`
          which is non-destructive — it fills empty slots without
          overwriting translator edits. The user-supplied `--method`
          flag only applies to the first (create) path, because Mode 2
          must not risk wiping existing translations.
    """
    try:
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        corpus = BilingualCorpus.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load {corpus_path.name}: {e}")
        return

    units = converter.to_units(corpus)
    if not units:
        logger.warning(f"{corpus_path.name}: no translatable units, skipping")
        return

    # Slug encodes the mod namespace so the same loc-file stem across
    # multiple mods lands in distinct Weblate components. e.g.
    # `1122837889-more-traits-XComGame` vs `base-xcom2-wotc-XComGame`.
    slug = f"{corpus.namespace}-{corpus_path.stem}"

    component = client.get_component(slug)
    if component is None:
        _corpus_upload_mode_create(
            client,
            slug,
            corpus,
            units,
            target_lang,
            method,
            license,
            license_url,
        )
    else:
        _corpus_upload_mode_incremental(
            client, slug, component, corpus, units, target_lang, method, yes
        )


def _corpus_upload_mode_create(
    client: WeblateClient,
    slug: str,
    corpus: BilingualCorpus,
    units: list[tuple[str, str, str, str]],
    target_lang: str,
    method: str,
    license: str,
    license_url: str,
) -> None:
    """Mode 1 — first-time upload: create component + bulk load.

    `manage_units=True` + `edit_template=True` are required so later
    Mode 2 incremental updates can POST `create_unit` to add new source
    strings. Verified against hosted.weblate.org 2026-04: without both
    flags, create_unit on a non-glossary bilingual CSV component returns
    HTTP 403 "Adding strings is disabled in the component configuration".
    """
    source_csv = _units_to_source_csv_bytes(units)
    translated_csv = _units_to_translation_csv_bytes(units)

    logger.info(f"Creating component '{slug}' ({len(units)} units)")
    client.create_component(
        name=slug,
        slug=slug,
        csv_bytes=source_csv,
        source_language=corpus.source_lang,
        license=license,
        license_url=license_url,
        manage_units=True,
        edit_template=True,
    )
    client.create_translation(slug, target_lang)

    if any(tgt for _, _, tgt, _ in units):
        client.upload_file(slug, target_lang, translated_csv, method=method)


def _corpus_upload_mode_incremental(
    client: WeblateClient,
    slug: str,
    component: dict[str, Any],
    corpus: BilingualCorpus,
    units: list[tuple[str, str, str, str]],
    target_lang: str,
    method: str,
    yes: bool,
) -> None:
    """Mode 2 — component already exists: add only new units, fill empty
    translations non-destructively.

    `method` only matters for the translation-side upload; the default
    `translate` mode is forced internally for safety. If the user
    explicitly passes `--method replace` we respect it (with the usual
    confirmation prompt) — that is the escape hatch for "I really do
    want to blow away existing target translations".
    """
    # Diff source-language contexts: anything local that Weblate doesn't
    # know about yet must be POSTed via create_unit.
    existing_contexts: set[str] = set()
    try:
        existing_contexts = {
            str(u.get("context", ""))
            for u in client.list_units(slug, corpus.source_lang)
        }
    except WeblateAPIError as e:
        logger.warning(
            f"Could not list existing units for '{slug}' ({e}); assuming "
            "none exist. Duplicate-context errors may follow."
        )

    new_units = [u for u in units if u[0] not in existing_contexts]
    existing_count = len(units) - len(new_units)

    logger.info(
        f"Component '{slug}' exists: {existing_count} already present, "
        f"{len(new_units)} new to add"
    )

    if new_units:
        added = 0
        for context, source, _target, _note in new_units:
            # Non-glossary bilingual CSV expects monolingual-template shape:
            # `key` = compound_key (used as both the internal id and the
            # context), `value` = source string wrapped in a list (Weblate
            # treats source/target as plural-form list by default). The
            # target for the new unit arrives empty on zh_Hans side and is
            # filled by the subsequent upload_file(method=translate) call.
            # Verified against hosted.weblate.org 2026-04.
            body: dict[str, Any] = {
                "key": context,
                "value": [source],
            }
            try:
                client.create_unit(slug, corpus.source_lang, body)
                added += 1
            except WeblateAPIError as e:
                logger.warning(f"create_unit {context!r} failed: {e}")
        logger.info(f"Added {added}/{len(new_units)} new units to '{slug}'")

    client.create_translation(slug, target_lang)

    # Target-side update. Default to `translate` mode in Mode 2 — it only
    # fills empty slots and never overwrites translator work. Honor an
    # explicit `--method replace` only after user confirmation.
    effective_method = method
    if method == "replace":
        unit_count = component.get("stats", {}).get("total", "?")
        logger.warning(
            f"Component '{slug}' already has {unit_count} units. "
            "--method=replace will overwrite existing translations."
        )
        if not yes and not typer.confirm("Continue?", default=False):
            raise typer.Exit(1)
    else:
        effective_method = "translate"

    if any(tgt for _, _, tgt, _ in units):
        translated_csv = _units_to_translation_csv_bytes(units)
        client.upload_file(slug, target_lang, translated_csv, method=effective_method)


def _upload_glossary(
    client: WeblateClient,
    glossary_path: Path,
    namespace: str,
    target_lang: str,
    method: str,
    license: str = "",
    license_url: str = "",
) -> None:
    """Upload a P2 glossary CSV as a per-mod Weblate glossary component.

    Slug convention: `glossary-{namespace}`. Each mod (and the base game)
    gets its own glossary component, so uploads never overwrite another
    mod's work and Weblate's project-level glossary linking gives
    translators the combined hint set automatically.

    Behavior:
        - **Mode 1 (fresh)**: component missing → `create_component` with
          the full source CSV, then upload target CSV, then apply flags.
        - **Mode 2 (incremental)**: component exists → fetch existing
          contexts, POST `create_unit` for new ones only, then refresh
          flags. No destructive upload_file calls.
    """
    rows = _load_glossary_rows(glossary_path)
    if not rows:
        logger.warning(f"Glossary {glossary_path.name} is empty, skipping")
        return

    slug = f"glossary-{namespace}"
    component = client.get_component(slug)
    if component is not None and component.get("file_format") != "csv":
        # Weblate auto-creates a TBX-format glossary on first component
        # insertion. Delete and recreate as CSV so our upload shape matches.
        logger.warning(
            f"Existing glossary component '{slug}' uses file_format="
            f"{component.get('file_format')!r}; deleting and recreating as csv"
        )
        client.delete_component(slug)
        component = None

    if component is None:
        _glossary_upload_mode_create(
            client,
            slug,
            rows,
            target_lang,
            method,
            license,
            license_url,
        )
    else:
        _glossary_upload_mode_incremental(client, slug, rows, target_lang)

    # Source language is usually "en" but we read it from the component
    # metadata after creation/update because `extra_flags` can only be set
    # on source-language units.
    component = client.get_component(slug) or {}
    source_lang_field = component.get("source_language") or {}
    source_lang = source_lang_field.get("code", "en")
    _mark_glossary_flags(client, slug, source_lang, target_lang, rows)


def _load_glossary_rows(glossary_path: Path) -> list[dict[str, str]]:
    """Read a P2 glossary CSV into a list of normalized dict rows."""
    with glossary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _glossary_rows_to_csvs(
    rows: list[dict[str, str]],
) -> tuple[bytes, bytes]:
    """Split glossary rows into (source CSV, translation CSV) bytes.

    Weblate's `csv` file format treats the `target` column as "current
    language content", so the source CSV must carry English in `target`
    (Weblate reads it as the en-language file) while the translation
    CSV carries the target-language text. Rows without a source are
    dropped; rows without a target skip the translation CSV.
    """
    source_buf = io.StringIO()
    source_writer = _make_csv_writer(source_buf)
    source_writer.writeheader()
    translation_buf = io.StringIO()
    translation_writer = _make_csv_writer(translation_buf)
    translation_writer.writeheader()
    for row in rows:
        src = row.get("source") or ""
        tgt = row.get("target") or ""
        cat = row.get("category") or "term"
        if not src:
            continue
        context = f"{src}::{cat}"
        source_writer.writerow(
            {
                "context": context,
                "source": src,
                "target": src,  # en-file: target col holds English
                "developer_comments": cat,
            }
        )
        if tgt:
            translation_writer.writerow(
                {
                    "context": context,
                    "source": src,
                    "target": tgt,
                    "developer_comments": cat,
                }
            )
    return (
        source_buf.getvalue().encode("utf-8"),
        translation_buf.getvalue().encode("utf-8"),
    )


def _glossary_upload_mode_create(
    client: WeblateClient,
    slug: str,
    rows: list[dict[str, str]],
    target_lang: str,
    method: str,
    license: str,
    license_url: str,
) -> None:
    """Mode 1 — component missing: bulk-create via CSV docfile.

    `edit_template=True` is required so Mode 2 can later POST new terms
    via `create_unit`; glossary already defaults to `manage_units=True`
    but `edit_template` stays False unless we opt in. Verified against
    hosted.weblate.org 2026-04: without edit_template, create_unit on a
    glossary component also returns the 403 "Adding strings is disabled".
    """
    source_csv, translation_csv = _glossary_rows_to_csvs(rows)

    logger.info(f"Creating glossary component '{slug}' ({len(rows)} rows)")
    client.create_component(
        name=slug,
        slug=slug,
        csv_bytes=source_csv,
        is_glossary=True,
        license=license,
        license_url=license_url,
        edit_template=True,
    )
    client.create_translation(slug, target_lang)
    if translation_csv.count(b"\n") > 1:  # header + at least one row
        client.upload_file(slug, target_lang, translation_csv, method=method)


def _glossary_upload_mode_incremental(
    client: WeblateClient,
    slug: str,
    rows: list[dict[str, str]],
    target_lang: str,
) -> None:
    """Mode 2 — component exists: additive unit POSTs, no destructive upload.

    Mod updates and base-game term additions land here. We POST only
    contexts that Weblate doesn't know about yet; existing glossary
    terms (and any translator edits to them) are left untouched.

    Body shape note: glossary components in Weblate treat their CSV as
    a monolingual template — `create_unit` requires `{"key": context,
    "value": [source]}`, same as non-glossary bilingual CSV, **not** the
    glossary-flavored `{"source", "target", "context"}` shape that some
    older docs suggest. Target translations for freshly-POSTed units
    arrive as `[""]` and are filled by the subsequent
    `upload_file(method=translate)` call, which is additive and safe.

    `_mark_glossary_flags` runs afterwards to re-apply
    `do_not_translate` / `same_as_source` state — cheap idempotent
    PATCHes.
    """
    client.create_translation(slug, target_lang)

    existing_contexts: set[str] = set()
    try:
        existing_contexts = {
            str(u.get("context", "")) for u in client.list_units(slug, "en")
        }
    except WeblateAPIError as e:
        logger.warning(
            f"Could not list glossary units for '{slug}' ({e}); "
            "assuming none exist, duplicate-context errors may follow."
        )

    added = 0
    skipped = 0
    for row in rows:
        src = row.get("source") or ""
        cat = row.get("category") or "term"
        if not src:
            continue
        context = f"{src}::{cat}"
        if context in existing_contexts:
            skipped += 1
            continue
        body: dict[str, Any] = {
            "key": context,
            "value": [src],
        }
        try:
            client.create_unit(slug, "en", body)
            added += 1
        except WeblateAPIError as e:
            logger.warning(f"glossary create_unit {context!r} failed: {e}")

    logger.info(f"Glossary '{slug}': added={added}, existing={skipped}")

    # Push target-language translations. `method=translate` fills empty
    # slots (the newly-POSTed units) without overwriting existing
    # translator edits. This is the only path that actually writes the
    # target text for units we just created via create_unit.
    _, translation_csv = _glossary_rows_to_csvs(rows)
    if translation_csv.count(b"\n") > 1:
        client.upload_file(slug, target_lang, translation_csv, method="translate")


def _mark_glossary_flags(
    client: WeblateClient,
    slug: str,
    source_lang: str,
    target_lang: str,
    rows: list[dict[str, str]],
) -> None:
    """Apply P2 glossary flags to Weblate units.

    Weblate has two distinct concerns that must target different languages:
      1. `extra_flags="read-only"` — a SOURCE-STRING property; can only be
         PATCHed on units in the component's source language (e.g. `en`).
         Trying to set it on a target-language unit raises HTTP 403
         "Source strings properties can be set only on source strings".
      2. `state=10` (needs-editing) — a translation-unit property; must be
         PATCHed on TARGET-language units. Weblate re-validates the target
         on PATCH, so for plural units we must also pass the existing
         target back in list form to avoid a 400 "Number of plurals does
         not match" error.
    """
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        src = row.get("source") or ""
        cat = row.get("category") or "term"
        index[f"{src}::{cat}"] = row

    read_only_count = 0
    needs_editing_count = 0
    skipped = 0

    # Pass 1: do_not_translate → extra_flags on SOURCE-language units
    for unit in client.list_units(slug, source_lang):
        row = index.get(unit.get("context", ""))
        if row is None or row.get("do_not_translate") != "true":
            continue
        try:
            client.patch_unit(int(unit["id"]), {"extra_flags": "read-only"})
            read_only_count += 1
        except WeblateAPIError as e:
            logger.warning(f"patch_unit {unit.get('id')} (read-only): {e}")
            skipped += 1

    # Pass 2: same_as_source → state=10 on TARGET-language units. Pass the
    # existing target list-form back so Weblate's plural re-validation passes.
    for unit in client.list_units(slug, target_lang):
        row = index.get(unit.get("context", ""))
        if row is None or row.get("same_as_source") != "true":
            continue
        if row.get("do_not_translate") == "true":
            # Already read-only — state change is not meaningful.
            continue
        target_value = unit.get("target", [])
        if not isinstance(target_value, list):
            target_value = [str(target_value)]
        try:
            client.patch_unit(
                int(unit["id"]),
                {"state": 10, "target": target_value},
            )
            needs_editing_count += 1
        except WeblateAPIError as e:
            logger.warning(f"patch_unit {unit.get('id')} (needs-editing): {e}")
            skipped += 1

    logger.info(
        f"Glossary flags applied: read-only={read_only_count}, "
        f"needs-editing={needs_editing_count}, skipped={skipped}"
    )


def _count_translated(
    source_file: LocalizationFile, translations: dict[str, str]
) -> int:
    """Count how many source entries have a translation in the CSV.

    Goes through the same `iter_compound_keys` helper as aligner and
    converter so the counter's key shape cannot drift from what writeback
    actually applies.
    """
    hits = 0
    for compound_key, entry, _section in iter_compound_keys(source_file):
        if entry.is_append and entry.struct_fields is not None:
            for field in entry.struct_fields:
                if field.key not in TRANSLATABLE_STRUCT_FIELDS:
                    continue
                if f"{compound_key}::{field.key}" in translations:
                    hits += 1
        elif compound_key in translations:
            hits += 1
    return hits


if __name__ == "__main__":
    app()
