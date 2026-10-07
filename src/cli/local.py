"""`x2loc local`: translate the Workshop mods already on this machine.

Mods are read straight from the local Workshop content directory, so no
SteamCMD login is involved, and every translated mod is installed as a
standalone overlay mod under the game's `XComGame/Mods` directory.
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from src.cli.batch import new_run_dir, open_pipeline, translate_item, write_summary
from src.config import LocalConfigSchema, ServiceConfigSchema
from src.core.mod_resolver import ModResolveError
from src.core.overlay_mod import install_overlay
from src.core.workshop import WorkshopInputError, load_workshop_item
from src.jobs.pipeline import prepare_works
from src.models._share import DEFAULT_LLM_CONCURRENCY, MAX_LLM_CONCURRENCY
from src.models.job import JobStatus
from src.models.workshop import (
    TARGET_LANGUAGE,
    WorkshopItemSchema,
    WorkshopLimitsSchema,
)


class LocalWorkshopSource:
    """Serve items that were scanned from disk before the run started."""

    def __init__(self, items: dict[str, WorkshopItemSchema]) -> None:
        self._items = items

    async def download(self, workshop_id: str) -> WorkshopItemSchema:
        return self._items[workshop_id]


def local(
    workshop_ids: Annotated[
        list[str] | None,
        typer.Argument(help="Workshop ids to translate; all local mods if omitted."),
    ] = None,
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Service TOML.")
    ] = Path("configs/weblate.local.toml"),
    workshop_dir: Annotated[
        Path | None,
        typer.Option("--workshop-dir", help="Overrides [local] workshop_dir."),
    ] = None,
    mods_dir: Annotated[
        Path | None, typer.Option("--mods-dir", help="Overrides [local] mods_dir.")
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for run folders.")
    ] = Path("output/local"),
    llm_concurrency: Annotated[
        int,
        typer.Option(
            "--llm-concurrency",
            min=1,
            max=MAX_LLM_CONCURRENCY,
            help="LLM requests in flight per job.",
        ),
    ] = DEFAULT_LLM_CONCURRENCY,
) -> None:
    """Translate local Workshop mods and install them as overlay mods."""
    config = ServiceConfigSchema.from_toml(config_path)
    config = config.model_copy(update={"data_root": config.data_root / "local"})
    paths = _resolve_paths(config.local, workshop_dir, mods_dir)
    items, skipped = _load_items(paths.workshop_dir, workshop_ids, config.limits)
    logger.info("{} local mods to translate, {} skipped", len(items), len(skipped))
    run_dir = new_run_dir(output)
    results = skipped
    if items:
        results += asyncio.run(
            _run(
                config,
                items,
                mods_dir=paths.mods_dir,
                run_dir=run_dir,
                llm_concurrency=llm_concurrency,
            )
        )
    write_summary(run_dir, results)


def _resolve_paths(
    configured: LocalConfigSchema | None,
    workshop_dir: Path | None,
    mods_dir: Path | None,
) -> LocalConfigSchema:
    if workshop_dir is None or mods_dir is None:
        if configured is None:
            raise typer.BadParameter(
                "set [local] workshop_dir and mods_dir in the TOML, "
                "or pass --workshop-dir and --mods-dir"
            )
        workshop_dir = workshop_dir or configured.workshop_dir
        mods_dir = mods_dir or configured.mods_dir
    for path in (workshop_dir, mods_dir):
        if not path.is_dir():
            raise typer.BadParameter(f"directory does not exist: {path}")
    return LocalConfigSchema(workshop_dir=workshop_dir, mods_dir=mods_dir)


def _load_items(
    workshop_dir: Path, wanted: list[str] | None, limits: WorkshopLimitsSchema
) -> tuple[dict[str, WorkshopItemSchema], list[dict[str, object]]]:
    """Scan the local mods up front; mods with nothing to translate are skipped.

    Every source file is parsed here and again by the job; the repeat costs
    seconds even for the largest mods.
    """
    available = {
        path.name: path
        for path in sorted(workshop_dir.iterdir())
        if path.is_dir() and path.name.isdecimal()
    }
    if wanted:
        missing = sorted(set(wanted) - available.keys())
        if missing:
            raise typer.BadParameter(
                f"not found in {workshop_dir}: {', '.join(missing)}"
            )
        available = {workshop_id: available[workshop_id] for workshop_id in wanted}

    items: dict[str, WorkshopItemSchema] = {}
    skipped: list[dict[str, object]] = []
    for workshop_id, mod_root in available.items():
        try:
            item = load_workshop_item(mod_root, workshop_id, limits)
            prepare_works(item, TARGET_LANGUAGE)
        except (WorkshopInputError, ModResolveError) as exc:
            logger.warning("[{}] skipped: {}", workshop_id, exc)
            skipped.append(
                {"workshop_id": workshop_id, "status": "skipped", "reason": str(exc)}
            )
            continue
        items[workshop_id] = item
    return items, skipped


async def _run(
    config: ServiceConfigSchema,
    items: dict[str, WorkshopItemSchema],
    *,
    mods_dir: Path,
    run_dir: Path,
    llm_concurrency: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    async with open_pipeline(config, LocalWorkshopSource(items)) as (
        manager,
        pipeline,
    ):
        for workshop_id, item in items.items():
            title = item.mod_info.mod_title
            result = await translate_item(
                manager,
                pipeline,
                workshop_id=workshop_id,
                title=title,
                run_dir=run_dir,
                llm_concurrency=llm_concurrency,
            )
            if result["status"] == JobStatus.SUCCEEDED:
                installed = await asyncio.to_thread(
                    install_overlay,
                    artifact=run_dir / f"{workshop_id}.zip",
                    mods_dir=mods_dir,
                    workshop_id=workshop_id,
                    title=title,
                )
                logger.success("[{}] installed to {}", workshop_id, installed)
                result["installed"] = str(installed)
            results.append(result)
    return results
