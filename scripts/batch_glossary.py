"""Stage 3: extract and upload two separate glossaries to Weblate.

Two glossary components serve different roles so translators can
immediately distinguish official translations from community terms:

    glossary-base-xcom2-wotc    Base game terms (Firaxis official)
    glossary-mods               Community mod terms (cross-mod dedup)

The mod glossary excludes source strings that already appear in the
base game glossary, so there is zero term overlap between the two
components. Translators see hints from BOTH glossaries simultaneously
(Weblate's project-level glossary aggregation) but the component name
tells them which is authoritative.

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
from src.models.glossary import Glossary  # noqa: E402
from src.models.weblate import WeblateConfigSchema  # noqa: E402
from src.services.weblate import WeblateClient  # noqa: E402

BASE_CORPUS_DIR: Path = ROOT / "output" / "corpus_game" / "_base"
MOD_CORPUS_ROOT: Path = ROOT / "output" / "corpus"

OUTPUT_BASE_CSV: Path = ROOT / "output" / "glossary_base.csv"
OUTPUT_MODS_CSV: Path = ROOT / "output" / "glossary_mods.csv"
CONFIG_TOML: Path = ROOT / "configs" / "weblate.toml"

# Weblate component slugs via the `glossary-{namespace}` convention.
BASE_GLOSSARY_NAMESPACE: str = "base-xcom2-wotc"  # → slug: glossary-base-xcom2-wotc
MODS_GLOSSARY_NAMESPACE: str = "mods"  # → slug: glossary-mods


def _load_corpora_from_dir(corpus_dir: Path) -> list[BilingualCorpus]:
    corpora: list[BilingualCorpus] = []
    for jf in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            corpora.append(BilingualCorpus.model_validate(data))
        except Exception as e:
            logger.warning(f"Failed to load {jf}: {e}")
    return corpora


def _load_mod_corpora(root: Path) -> list[BilingualCorpus]:
    corpora: list[BilingualCorpus] = []
    for namespace_dir in sorted(root.iterdir()):
        if not namespace_dir.is_dir() or namespace_dir.name.startswith("_"):
            continue
        corpora.extend(_load_corpora_from_dir(namespace_dir))
    return corpora


def _dedup_mod_glossary(base: Glossary, mod: Glossary) -> Glossary:
    """Remove mod terms whose source already appears in the base glossary.

    This ensures zero overlap between the two Weblate glossary
    components: every source string lives in exactly one of them.
    Translators see hints from both via project-level aggregation, but
    a given source text resolves to either the official or the community
    translation, never a confusing duplicate.
    """
    base_sources = {t.source for t in base.terms}
    filtered = [t for t in mod.terms if t.source not in base_sources]
    logger.info(
        f"Mod glossary dedup: {len(mod.terms)} total → {len(filtered)} "
        f"unique (removed {len(mod.terms) - len(filtered)} base overlaps)"
    )
    return Glossary(
        source_lang=mod.source_lang,
        target_lang=mod.target_lang,
        terms=filtered,
    )


def _preview(name: str, glossary: Glossary) -> None:
    total = len(glossary.terms)
    empty = sum(1 for t in glossary.terms if not t.target)
    same = sum(1 for t in glossary.terms if t.same_as_source)
    real = sum(1 for t in glossary.terms if t.target and not t.same_as_source)
    categories: dict[str, int] = {}
    for t in glossary.terms:
        categories[t.category] = categories.get(t.category, 0) + 1

    print(f"\n{name} summary:")
    print(f"  total terms:        {total}")
    print(f"  real translations:  {real}")
    print(f"  same as source:     {same}")
    print(f"  empty target:       {empty}")
    print(f"  categories ({len(categories)} unique):")
    for cat, count in sorted(categories.items(), key=lambda kv: -kv[1])[:10]:
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

    print("Stage 3: dual glossary extraction + upload")
    print(f"  base corpora:  {BASE_CORPUS_DIR}")
    print(f"  mod corpora:   {MOD_CORPUS_ROOT}")
    print()

    if not BASE_CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"Missing base corpus dir: {BASE_CORPUS_DIR}")
    if not MOD_CORPUS_ROOT.is_dir():
        raise FileNotFoundError(f"Missing mod corpus root: {MOD_CORPUS_ROOT}")

    extractor = TermExtractor()
    writer = GlossaryWriter()

    # Extract base game terms
    base_corpora = _load_corpora_from_dir(BASE_CORPUS_DIR)
    print(f"Loaded {len(base_corpora)} base-game corpora")
    base_glossary = extractor.extract(base_corpora)
    print(f"  → {len(base_glossary.terms)} base-game terms")

    # Extract mod terms + dedup against base
    mod_corpora = _load_mod_corpora(MOD_CORPUS_ROOT)
    print(f"Loaded {len(mod_corpora)} mod corpora")
    raw_mod_glossary = extractor.extract(mod_corpora)
    print(f"  → {len(raw_mod_glossary.terms)} mod terms (pre-dedup)")
    mod_glossary = _dedup_mod_glossary(base_glossary, raw_mod_glossary)

    _preview("glossary-base-xcom2-wotc", base_glossary)
    _preview("glossary-mods", mod_glossary)

    # Write local CSVs
    writer.write_csv(base_glossary, OUTPUT_BASE_CSV)
    writer.write_csv(mod_glossary, OUTPUT_MODS_CSV)
    print(f"\nWrote {OUTPUT_BASE_CSV}")
    print(f"Wrote {OUTPUT_MODS_CSV}")

    # Upload both
    cfg = _load_config()
    with WeblateClient(cfg) as client:
        print(f"\nUploading glossary-{BASE_GLOSSARY_NAMESPACE}...")
        _upload_glossary(
            client,
            OUTPUT_BASE_CSV,
            BASE_GLOSSARY_NAMESPACE,
            "zh_Hans",
            "replace",
            cfg.license,
            cfg.license_url,
        )

        print(f"\nUploading glossary-{MODS_GLOSSARY_NAMESPACE}...")
        _upload_glossary(
            client,
            OUTPUT_MODS_CSV,
            MODS_GLOSSARY_NAMESPACE,
            "zh_Hans",
            "replace",
            cfg.license,
            cfg.license_url,
        )

    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
