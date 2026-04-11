"""Stage 3: extract a consolidated project-wide glossary and upload it.

Merges `output/corpus_game/_base/` (base-game corpora, authoritative) and
`output/corpus/{namespace}/` (all mod corpora, community-translated) into
a single Weblate glossary component slug `glossary-x2loc`.

Consolidation policy (user-confirmed 2026-04-11):
    1. Coverage: base game + all mods — base game terms are the gold
       standard, mod terms fill gaps where base game doesn't have a
       matching source string.
    2. Dedup key: source text only (not source+target).
    3. Priority: base game wins. When base has a term with a non-empty
       target, that target is kept. When base's target is empty and a
       mod has a non-empty target, the mod's target fills the gap
       (better than nothing). When neither has a target, the row stays
       empty and waits for a translator to fill it.
    4. Extractor dedup operates on (source, target) tuples, so calling
       `TermExtractor.extract(all_corpora)` once would leave duplicate
       rows for "same source, different target" situations. We extract
       base and mods separately and then merge by source in a dedicated
       pass to get the base-wins semantics.

Run:
    uv run python scripts/batch_glossary.py
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.cli.app import _upload_glossary  # noqa: E402
from src.core.extractor import TermExtractor  # noqa: E402
from src.export.writer import GlossaryWriter  # noqa: E402
from src.models.corpus import BilingualCorpus  # noqa: E402
from src.models.glossary import Glossary, GlossaryTerm  # noqa: E402
from src.models.weblate import WeblateConfigSchema  # noqa: E402
from src.services.weblate import WeblateClient  # noqa: E402

BASE_CORPUS_DIR: Path = ROOT / "output" / "corpus_game" / "_base"
MOD_CORPUS_ROOT: Path = ROOT / "output" / "corpus"
OUTPUT_CSV: Path = ROOT / "output" / "glossary_x2loc.csv"
OUTPUT_JSON: Path = ROOT / "output" / "glossary_x2loc.json"
CONFIG_TOML: Path = ROOT / "configs" / "weblate.toml"

# Target slug on Weblate. Stored as the `namespace` arg to `_upload_glossary`
# so the final slug becomes `glossary-x2loc`.
PROJECT_GLOSSARY_NAMESPACE: str = "x2loc"


def _load_corpora_from_dir(corpus_dir: Path) -> list[BilingualCorpus]:
    """Load every BilingualCorpus JSON directly under `corpus_dir`."""
    corpora: list[BilingualCorpus] = []
    for jf in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            corpora.append(BilingualCorpus.model_validate(data))
        except Exception as e:
            logger.warning(f"Failed to load {jf}: {e}")
    return corpora


def _load_mod_corpora(root: Path) -> list[BilingualCorpus]:
    """Load every mod corpus under `root/{namespace}/*.json`.

    Skips the `_base/` subdirectory to avoid double-counting base game
    corpora that happened to land there in an earlier run.
    """
    corpora: list[BilingualCorpus] = []
    for namespace_dir in sorted(root.iterdir()):
        if not namespace_dir.is_dir() or namespace_dir.name.startswith("_"):
            continue
        corpora.extend(_load_corpora_from_dir(namespace_dir))
    return corpora


def _consolidate(base: Glossary, mod: Glossary) -> Glossary:
    """Merge mod glossary into base glossary with base-wins semantics.

    Order of precedence for a given source text:
        1. Base game term with non-empty target — kept verbatim.
        2. Base game term with empty target + mod term with non-empty
           target — base metadata (category, contexts) is preserved,
           but the empty target is replaced by the mod's target.
        3. Base game term only, both targets empty — kept as-is.
        4. Mod term whose source never appears in base — added as-is.
    """
    by_source: dict[str, GlossaryTerm] = {t.source: t for t in base.terms}

    overrides = 0
    additions = 0

    for mod_term in mod.terms:
        if mod_term.source not in by_source:
            by_source[mod_term.source] = mod_term
            additions += 1
            continue

        existing = by_source[mod_term.source]
        if not existing.target and mod_term.target:
            merged = GlossaryTerm(
                source=existing.source,
                target=mod_term.target,
                category=existing.category,
                do_not_translate=existing.do_not_translate,
                same_as_source=mod_term.same_as_source,
                contexts=[*existing.contexts, *mod_term.contexts],
            )
            by_source[existing.source] = merged
            overrides += 1

    logger.info(
        f"Consolidation: base_terms={len(base.terms)}, "
        f"mod_terms={len(mod.terms)}, "
        f"additions={additions}, target_overrides={overrides}, "
        f"final={len(by_source)}"
    )

    terms = sorted(by_source.values(), key=lambda t: (t.category, t.source))
    return Glossary(
        source_lang=base.source_lang,
        target_lang=base.target_lang,
        terms=terms,
    )


def _preview(glossary: Glossary) -> None:
    """Print quick stats before uploading."""
    total = len(glossary.terms)
    empty = sum(1 for t in glossary.terms if not t.target)
    same_as_source = sum(1 for t in glossary.terms if t.same_as_source)
    real_translation = sum(
        1 for t in glossary.terms if t.target and not t.same_as_source
    )
    categories: dict[str, int] = {}
    for t in glossary.terms:
        categories[t.category] = categories.get(t.category, 0) + 1

    print()
    print("glossary-x2loc summary:")
    print(f"  total terms:        {total}")
    print(f"  real translations:  {real_translation}")
    print(f"  same as source:     {same_as_source}")
    print(f"  empty target:       {empty}")
    print(f"  categories ({len(categories)} unique):")
    for cat, count in sorted(categories.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {cat:<25} {count}")


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


def main() -> None:
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level: <7} | {message}",
    )

    print("Stage 3: consolidated glossary extraction + upload")
    print(f"  base corpora:  {BASE_CORPUS_DIR}")
    print(f"  mod corpora:   {MOD_CORPUS_ROOT}")
    print()

    if not BASE_CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"Missing base corpus dir: {BASE_CORPUS_DIR}")
    if not MOD_CORPUS_ROOT.is_dir():
        raise FileNotFoundError(f"Missing mod corpus root: {MOD_CORPUS_ROOT}")

    extractor = TermExtractor()
    writer = GlossaryWriter()

    # Stage 3a: extract base game terms (authoritative set)
    base_corpora = _load_corpora_from_dir(BASE_CORPUS_DIR)
    print(f"Loaded {len(base_corpora)} base-game corpora")
    base_glossary = extractor.extract(base_corpora)
    print(f"  → {len(base_glossary.terms)} base-game terms")

    # Stage 3b: extract mod terms (community set)
    mod_corpora = _load_mod_corpora(MOD_CORPUS_ROOT)
    print(f"Loaded {len(mod_corpora)} mod corpora")
    mod_glossary = extractor.extract(mod_corpora)
    print(f"  → {len(mod_glossary.terms)} mod terms (pre-consolidation)")

    # Stage 3c: consolidate with base-wins semantics
    final = _consolidate(base_glossary, mod_glossary)
    _preview(final)

    # Write local artifacts for review / re-run
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    writer.write_csv(final, OUTPUT_CSV)
    writer.write_json(final, OUTPUT_JSON)
    print()
    print(f"Wrote CSV: {OUTPUT_CSV}")
    print(f"Wrote JSON: {OUTPUT_JSON}")

    # Stage 3d: upload to Weblate
    print()
    print(f"Uploading as 'glossary-{PROJECT_GLOSSARY_NAMESPACE}' to Weblate...")
    cfg = _load_config()
    with WeblateClient(cfg) as client:
        _upload_glossary(
            client,
            OUTPUT_CSV,
            PROJECT_GLOSSARY_NAMESPACE,
            "zh_Hans",
            "replace",
            cfg.license,
            cfg.license_url,
        )

    print()
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
