"""`x2loc batch`: translate every mod of a Workshop collection in-process.

Glossaries come from the persistent local store (one full read, then
incremental syncs) and newly mined custom terms are queued locally and
published once when the run ends. A run that stops early leaves the queue
on disk; the next run publishes it.
"""

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from httpx2 import AsyncClient
from loguru import logger

from src.agent.llm import build_llm_http_client
from src.config import ServiceConfigSchema
from src.core.workshop import parse_workshop_url
from src.jobs._share import GLOSSARY_TTL_SECONDS
from src.jobs.manager import JobManager
from src.jobs.pipeline import WorkshopPipeline, reset_work_dirs
from src.models.job import JobRecordSchema, JobStatus, WorkshopJobRequestSchema
from src.models.workshop import TARGET_LANGUAGE, XCOM2_APP_ID, WorkshopMetadataSchema
from src.services.glossary import validate_weblate_components
from src.services.glossary_store import DeferredGlossaryWriter, LocalGlossaryStore
from src.services.steam import (
    STEAM_RESULT_OK,
    SteamDownloader,
    fetch_collection_items,
)
from src.services.weblate import AsyncWeblateClient


def batch(
    collection_url: Annotated[
        str, typer.Argument(help="Steam Workshop collection URL.")
    ],
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Service TOML.")
    ] = Path("configs/weblate.local.toml"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for run folders.")
    ] = Path("output/batch"),
    max_size_mb: Annotated[
        float | None,
        typer.Option("--max-size-mb", help="Skip items larger than this."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Translate at most N items.")
    ] = None,
) -> None:
    """Translate a Workshop collection; publish new glossary terms at the end."""
    config = ServiceConfigSchema.from_toml(config_path)
    # A private work tree: the service in the same container resets its own.
    config = config.model_copy(update={"data_root": config.data_root / "batch"})
    run_dir = output / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    asyncio.run(
        _run(
            config,
            collection_id=parse_workshop_url(collection_url),
            run_dir=run_dir,
            max_bytes=None if max_size_mb is None else int(max_size_mb * 1_000_000),
            limit=limit,
        )
    )


async def _run(
    config: ServiceConfigSchema,
    *,
    collection_id: str,
    run_dir: Path,
    max_bytes: int | None,
    limit: int | None,
) -> None:
    async with AsyncClient(timeout=60.0, trust_env=False) as steam_web:
        items = await fetch_collection_items(collection_id, client=steam_web)
    selected = _select(items, max_bytes=max_bytes, limit=limit)
    logger.info(
        "Collection {}: {} items, {} selected", collection_id, len(items), len(selected)
    )

    reset_work_dirs(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    llm_client = build_llm_http_client()
    try:
        async with AsyncWeblateClient(config.weblate) as weblate:
            await validate_weblate_components(weblate, config.glossary)
            store = LocalGlossaryStore(
                weblate, config.glossary_cache_dir, refresh_seconds=GLOSSARY_TTL_SECONDS
            )
            writer = DeferredGlossaryWriter(
                store,
                component_slug=config.glossary.custom_slug,
                target_lang=TARGET_LANGUAGE,
            )
            manager = JobManager()
            pipeline = WorkshopPipeline(
                config=config,
                jobs=manager,
                steam=SteamDownloader(
                    executable=config.steam.executable,
                    steam_root=config.steam.root,
                    username=config.steam.steam_username,
                    password=config.steam.steam_password,
                    limits=config.limits,
                ),
                weblate=weblate,
                glossary_writer=writer,
                glossaries=store,
                llm_client=llm_client,
            )
            try:
                results = [
                    await _translate(manager, pipeline, item, run_dir)
                    for item in selected
                ]
            finally:
                await manager.close()
            published = await writer.flush(weblate)
            logger.success("Published {} new custom glossary terms", published)
    finally:
        await llm_client.aclose()

    summary = run_dir / "summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    counts = {
        status: sum(r["status"] == status for r in results) for status in JobStatus
    }
    logger.success(
        "Batch done: {}; summary at {}",
        ", ".join(f"{status.value} {n}" for status, n in counts.items() if n),
        summary,
    )


def _select(
    items: list[WorkshopMetadataSchema], *, max_bytes: int | None, limit: int | None
) -> list[WorkshopMetadataSchema]:
    usable = [
        item
        for item in items
        if item.result == STEAM_RESULT_OK and item.consumer_app_id == XCOM2_APP_ID
    ]
    skipped = len(items) - len(usable)
    if skipped:
        logger.warning("Skipping {} hidden, deleted or non-XCOM 2 items", skipped)
    if max_bytes is not None:
        usable = [item for item in usable if item.file_size <= max_bytes]
    return usable[:limit]


async def _translate(
    manager: JobManager,
    pipeline: WorkshopPipeline,
    item: WorkshopMetadataSchema,
    run_dir: Path,
) -> dict[str, object]:
    request = WorkshopJobRequestSchema.model_validate(
        {
            "workshop_url": "https://steamcommunity.com/sharedfiles/filedetails/"
            f"?id={item.publishedfileid}"
        }
    )
    submitted = manager.submit(request, title=item.title, runner=pipeline.run)
    record: JobRecordSchema = submitted
    stage = None
    async for record in manager.events(submitted.id):
        if record.stage != stage and record.stage is not None:
            stage = record.stage
            logger.info("[{}] {}: {}", item.publishedfileid, item.title, stage.value)
    artifact = None
    if record.artifact is not None:
        artifact = run_dir / f"{item.publishedfileid}.zip"
        await asyncio.to_thread(shutil.copy2, record.artifact.path, artifact)
    logger.info(
        "[{}] {} -> {}{}",
        item.publishedfileid,
        item.title,
        record.status.value,
        f" ({record.error_code})" if record.error_code else "",
    )
    return {
        "workshop_id": item.publishedfileid,
        "title": item.title,
        "status": record.status.value,
        "error_code": record.error_code,
        "units_total": record.progress.units_total,
        "units_translated": record.progress.units_translated,
        "terms_added": record.progress.terms_added,
        "artifact": str(artifact) if artifact else None,
    }
