"""Translation runs: a set of local mods translated and installed together.

A run opens the same pipeline as `x2loc local`, so it translates, writes
and installs exactly as the CLI does. One run at a time keeps the LLM
ceiling the settings name.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from loguru import logger

from src.cli.batch import open_pipeline
from src.cli.local import LocalWorkshopSource, load_items
from src.config import ServiceConfigSchema
from src.core.overlay_mod import install_overlay
from src.jobs.manager import JobManager
from src.jobs.pipeline import WorkshopPipeline, error_code
from src.models.job import (
    TERMINAL_STATUSES,
    JobRecordSchema,
    JobStatus,
    WorkshopJobRequestSchema,
)
from src.models.workshop import TARGET_LANGUAGE, WorkshopItemSchema
from src.services.stores import UploadReport, upload_local


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunError(Exception):
    """A run that cannot start; `code` names why."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class Run:
    id: str
    workshop_ids: list[str]
    upload_after: bool
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    # Mods that could not be parsed, by Workshop id, with their reason.
    skipped: dict[str, str] = field(default_factory=dict)
    # Job id of each submitted mod, by Workshop id.
    jobs: dict[str, str] = field(default_factory=dict)
    # Final job records, kept once the run's job manager is gone.
    records: dict[str, JobRecordSchema] = field(default_factory=dict)
    installed: dict[str, str] = field(default_factory=dict)
    upload: UploadReport | None = None
    error_code: str | None = None


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._active: tuple[Run, asyncio.Task[None]] | None = None
        self._manager: JobManager | None = None

    def runs(self) -> list[Run]:
        return sorted(self._runs.values(), key=lambda run: run.started_at, reverse=True)

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def start(
        self,
        config: ServiceConfigSchema,
        workshop_ids: list[str],
        *,
        jobs: int,
        llm_concurrency: int,
        upload_after: bool,
    ) -> Run:
        if self._active is not None:
            raise RunError("run_in_progress")
        if config.local is None:
            raise RunError("paths_not_set")
        if not workshop_ids:
            raise RunError("no_mods_selected")
        if not all((config.local.workshop_dir / wid).is_dir() for wid in workshop_ids):
            raise RunError("mod_not_found")
        run = Run(
            id=uuid4().hex[:12], workshop_ids=workshop_ids, upload_after=upload_after
        )
        self._runs[run.id] = run
        task = asyncio.create_task(
            self._run(run, config, jobs=jobs, llm_concurrency=llm_concurrency)
        )
        self._active = (run, task)
        task.add_done_callback(lambda _task: self._finish(run))
        return run

    def cancel(self, run_id: str) -> Run:
        """Cancel the active run; raises KeyError on an unknown id."""
        run = self._runs[run_id]
        if self._active is not None and self._active[0] is run:
            self._active[1].cancel()
        return run

    def job(self, job_id: str) -> JobRecordSchema | None:
        if self._manager is not None:
            record = self._manager.get(job_id)
            if record is not None:
                return record
        for run in self._runs.values():
            for record in run.records.values():
                if record.id == job_id:
                    return record
        return None

    def cancel_job(self, job_id: str) -> JobRecordSchema:
        if self._manager is None or self._manager.get(job_id) is None:
            raise KeyError(job_id)
        return self._manager.cancel(job_id)

    async def events(self, job_id: str) -> AsyncIterator[JobRecordSchema]:
        """Every update of a job until it ends; a finished one yields once."""
        if self._manager is not None and self._manager.get(job_id) is not None:
            async for record in self._manager.events(job_id):
                yield record
            return
        record = self.job(job_id)
        if record is None:
            raise KeyError(job_id)
        yield record

    async def aclose(self) -> None:
        if self._active is not None:
            task = self._active[1]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run(
        self,
        run: Run,
        config: ServiceConfigSchema,
        *,
        jobs: int,
        llm_concurrency: int,
    ) -> None:
        assert config.local is not None
        try:
            items, works, skipped = await asyncio.to_thread(
                load_items, config.local.workshop_dir, run.workshop_ids, config.limits
            )
            run.skipped = {
                str(entry["workshop_id"]): str(entry["reason"]) for entry in skipped
            }
            # The source hands each mod's works to its job and keeps none.
            slugs = {
                workshop_id: {work.asset.component_slug for work in mod_works}
                for workshop_id, mod_works in works.items()
            }
            source = LocalWorkshopSource(items, works)
            async with open_pipeline(config, source, job_concurrency=jobs) as (
                manager,
                pipeline,
            ):
                self._manager = manager
                await asyncio.gather(
                    *(
                        self._translate(
                            run,
                            config,
                            manager,
                            pipeline,
                            workshop_id,
                            item,
                            llm_concurrency=llm_concurrency,
                        )
                        for workshop_id, item in items.items()
                    )
                )
            self._manager = None
            translated = {
                slug
                for workshop_id, record in run.records.items()
                if record.status == JobStatus.SUCCEEDED
                for slug in slugs[workshop_id]
            }
            if (
                run.upload_after
                and not config.writes_weblate
                and config.weblate.token.get_secret_value()
                and translated
            ):
                run.upload = await upload_local(
                    config, language=TARGET_LANGUAGE, slugs=translated
                )
            failed = any(r.status == JobStatus.FAILED for r in run.records.values())
            run.status = RunStatus.FAILED if failed else RunStatus.SUCCEEDED
        except asyncio.CancelledError:
            run.status = RunStatus.CANCELLED
            raise
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error_code = error_code(exc)
            logger.exception("Run {} failed", run.id)
        finally:
            self._manager = None

    async def _translate(
        self,
        run: Run,
        config: ServiceConfigSchema,
        manager: JobManager,
        pipeline: WorkshopPipeline,
        workshop_id: str,
        item: WorkshopItemSchema,
        *,
        llm_concurrency: int,
    ) -> None:
        assert config.local is not None
        title = item.mod_info.mod_title
        request = WorkshopJobRequestSchema.model_validate(
            {
                "workshop_url": "https://steamcommunity.com/sharedfiles/filedetails/"
                f"?id={workshop_id}",
                "llm_concurrency": llm_concurrency,
            }
        )
        submitted = manager.submit(request, title=title, runner=pipeline.run)
        run.jobs[workshop_id] = submitted.id
        record = submitted
        async for record in manager.events(submitted.id):
            if record.status in TERMINAL_STATUSES:
                break
        run.records[workshop_id] = record
        if record.status == JobStatus.SUCCEEDED and record.artifact is not None:
            installed = await asyncio.to_thread(
                install_overlay,
                artifact=record.artifact.path,
                mods_dir=config.local.mods_dir,
                workshop_id=workshop_id,
                title=title,
            )
            run.installed[workshop_id] = str(installed)

    def _finish(self, run: Run) -> None:
        run.finished_at = datetime.now(UTC)
        self._active = None
