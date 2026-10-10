"""`x2loc batch`: translate every mod of a Workshop collection in-process.

Glossaries come from the persistent local store (one full read, then
incremental syncs) and newly mined custom terms are queued locally and
published once when the run ends. A run that stops early leaves the queue
on disk; the next run publishes it.
"""

import asyncio
import json
import math
import shutil
from collections import Counter
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from httpx2 import AsyncClient
from loguru import logger

from src.agent.transport import build_llm_http_client
from src.config import ServiceConfigSchema
from src.core.workshop import parse_workshop_url
from src.jobs._share import GLOSSARY_TTL_SECONDS, JOB_CONCURRENCY
from src.jobs.manager import JobManager
from src.jobs.pipeline import WorkshopPipeline, WorkshopSource, reset_work_dirs
from src.models._share import DEFAULT_LLM_CONCURRENCY, MAX_LLM_CONCURRENCY
from src.models.job import JobRecordSchema, WorkshopJobRequestSchema
from src.models.workshop import TARGET_LANGUAGE, XCOM2_APP_ID, WorkshopMetadataSchema
from src.services.context_index import ContextIndexSource
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
    """Translate a Workshop collection; publish new glossary terms at the end."""
    config = ServiceConfigSchema.from_toml(config_path)
    config = config.model_copy(update={"data_root": config.data_root / "batch"})
    asyncio.run(
        _run(
            config,
            collection_id=parse_workshop_url(collection_url),
            run_dir=new_run_dir(output),
            max_bytes=None if max_size_mb is None else int(max_size_mb * 1_000_000),
            limit=limit,
            llm_concurrency=llm_concurrency,
        )
    )


def new_run_dir(output: Path) -> Path:
    run_dir = output / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def _run(
    config: ServiceConfigSchema,
    *,
    collection_id: str,
    run_dir: Path,
    max_bytes: int | None,
    limit: int | None,
    llm_concurrency: int,
) -> None:
    async with AsyncClient(timeout=60.0, trust_env=False) as steam_web:
        items = await fetch_collection_items(collection_id, client=steam_web)
    selected = _select(items, max_bytes=max_bytes, limit=limit)
    logger.info(
        "Collection {}: {} items, {} selected", collection_id, len(items), len(selected)
    )

    source = SteamDownloader(
        executable=config.steam.executable,
        steam_root=config.steam.root,
        username=config.steam.steam_username,
        password=config.steam.steam_password,
        limits=config.limits,
    )
    async with open_pipeline(config, source) as (manager, pipeline):
        results = [
            await translate_item(
                manager,
                pipeline,
                workshop_id=item.publishedfileid,
                title=item.title,
                run_dir=run_dir,
                llm_concurrency=llm_concurrency,
            )
            for item in selected
        ]
    write_summary(run_dir, results)


@asynccontextmanager
async def open_pipeline(
    config: ServiceConfigSchema,
    source: WorkshopSource,
    *,
    job_concurrency: int = JOB_CONCURRENCY,
) -> AsyncGenerator[tuple[JobManager, WorkshopPipeline]]:
    """Wire one in-process pipeline over the persistent glossary store.

    Queued custom-glossary terms are published only when the body finishes
    without an exception; otherwise they stay on disk for the next run.
    """
    reset_work_dirs(config)
    llm_client = build_llm_http_client(config.agent.llm_timeout_seconds)
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
            manager = JobManager(job_concurrency)
            # A CLI run is short; the index it builds stays current through
            # the components the run syncs itself.
            context = ContextIndexSource(
                weblate, language=TARGET_LANGUAGE, ttl_seconds=math.inf
            )
            pipeline = WorkshopPipeline(
                config=config,
                jobs=manager,
                source=source,
                weblate=weblate,
                glossary_writer=writer,
                glossaries=store,
                context=context,
                llm_client=llm_client,
            )
            # Every job's term extraction reads the custom glossary; syncing it
            # while the jobs talk to Weblate keeps it off the tail of the run.
            warm = asyncio.create_task(
                store.units(config.glossary.custom_slug, TARGET_LANGUAGE)
            )
            try:
                yield manager, pipeline
            finally:
                await manager.close()
                await context.aclose()
                warm.cancel()
                await asyncio.gather(warm, return_exceptions=True)
            published = await writer.flush(weblate)
            logger.success("Published {} new custom glossary terms", published)
    finally:
        await llm_client.aclose()


def write_summary(run_dir: Path, results: list[dict[str, object]]) -> None:
    summary = run_dir / "summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    counts = Counter(str(result["status"]) for result in results)
    logger.success(
        "Run done: {}; summary at {}",
        ", ".join(f"{status} {n}" for status, n in counts.items()),
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


async def translate_item(
    manager: JobManager,
    pipeline: WorkshopPipeline,
    *,
    workshop_id: str,
    title: str,
    run_dir: Path,
    llm_concurrency: int,
) -> dict[str, object]:
    """Run one job to completion and copy its overlay zip into `run_dir`."""
    request = WorkshopJobRequestSchema.model_validate(
        {
            "workshop_url": "https://steamcommunity.com/sharedfiles/filedetails/"
            f"?id={workshop_id}",
            "llm_concurrency": llm_concurrency,
        }
    )
    submitted = manager.submit(request, title=title, runner=pipeline.run)
    record: JobRecordSchema = submitted
    stage = None
    async for record in manager.events(submitted.id):
        if record.stage != stage and record.stage is not None:
            stage = record.stage
            logger.info("[{}] {}: {}", workshop_id, title, stage.value)
    artifact = None
    if record.artifact is not None:
        artifact = run_dir / f"{workshop_id}.zip"
        await asyncio.to_thread(shutil.copy2, record.artifact.path, artifact)
    logger.info(
        "[{}] {} -> {}{}",
        workshop_id,
        title,
        record.status.value,
        f" ({record.error_code})" if record.error_code else "",
    )
    return {
        "workshop_id": workshop_id,
        "title": title,
        "status": record.status.value,
        "error_code": record.error_code,
        "units_total": record.progress.units_total,
        "units_translated": record.progress.units_translated,
        "units_untranslated": record.progress.units_untranslated,
        "terms_added": record.progress.terms_added,
        "artifact": str(artifact) if artifact else None,
    }
