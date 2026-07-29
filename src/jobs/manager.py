from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger

from src.jobs._share import JOB_CONCURRENCY
from src.models.job import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    JobRecordSchema,
    JobStatus,
    JobUpdateSchema,
    WorkshopJobRequestSchema,
)

type JobRunner = Callable[[str, WorkshopJobRequestSchema], Awaitable[None]]


class JobManager:
    """In-memory job table: records, dedup, cancellation and SSE fan-out.

    Records live for the process lifetime; a restart forgets them, and the
    runtime wipes the work directories at startup to match.
    """

    def __init__(self) -> None:
        self._records: dict[str, JobRecordSchema] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._active: dict[str, str] = {}
        self._subscribers: dict[str, set[asyncio.Queue[JobRecordSchema]]] = {}
        self._semaphore = asyncio.Semaphore(JOB_CONCURRENCY)

    def get(self, job_id: str) -> JobRecordSchema | None:
        return self._records.get(job_id)

    def submit(
        self,
        request: WorkshopJobRequestSchema,
        *,
        title: str,
        runner: JobRunner,
    ) -> JobRecordSchema:
        """Register and start a job; an active duplicate is returned as-is."""
        workshop_id = request.workshop_id
        existing_id = self._active.get(workshop_id)
        if existing_id is not None:
            return self._records[existing_id]

        record = JobRecordSchema(
            id=uuid4().hex[:12],
            workshop_id=workshop_id,
            workshop_url=str(request.workshop_url),
            title=title,
            target_lang=request.target_lang,
        )
        self._records[record.id] = record
        self._active[workshop_id] = record.id
        task = asyncio.create_task(self._execute(record.id, request, runner))
        self._tasks[record.id] = task
        task.add_done_callback(lambda _task: self._finalize(record.id, workshop_id))
        return record

    def cancel(self, job_id: str) -> JobRecordSchema:
        """Request cancellation of an active job; raises KeyError on unknown id.

        The returned record may still show the in-flight status: the terminal
        `cancelled` is written when the task actually unwinds, and reaches
        subscribers through the SSE stream.
        """
        record = self._records[job_id]
        if record.status in ACTIVE_STATUSES:
            self._tasks[job_id].cancel()
        return self._records[job_id]

    def update(self, job_id: str, patch: JobUpdateSchema) -> JobRecordSchema:
        updates: dict[str, object] = {
            name: getattr(patch, name) for name in patch.model_fields_set
        }
        updates["updated_at"] = datetime.now(UTC)
        record = self._records[job_id].model_copy(update=updates)
        if record.status in TERMINAL_STATUSES and record.finished_at is None:
            record = record.model_copy(update={"finished_at": datetime.now(UTC)})
        self._records[job_id] = record
        for queue in self._subscribers.get(job_id, ()):
            queue.put_nowait(record)
        return record

    async def events(self, job_id: str) -> AsyncIterator[JobRecordSchema]:
        """Yield the current snapshot, then every update until a terminal one.

        The queue is registered before the snapshot is read, so an update
        landing in between is delivered twice rather than lost; SSE events
        carry full state, so duplicates are harmless.
        """
        queue: asyncio.Queue[JobRecordSchema] = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(queue)
        try:
            record = self._records[job_id]
            yield record
            while record.status not in TERMINAL_STATUSES:
                record = await queue.get()
                yield record
        finally:
            self._subscribers[job_id].discard(queue)

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _execute(
        self, job_id: str, request: WorkshopJobRequestSchema, runner: JobRunner
    ) -> None:
        try:
            async with self._semaphore:
                await runner(job_id, request)
        except asyncio.CancelledError:
            self.update(job_id, JobUpdateSchema(status=JobStatus.CANCELLED, stage=None))
            raise

    def _finalize(self, job_id: str, workshop_id: str) -> None:
        self._tasks.pop(job_id, None)
        if self._active.get(workshop_id) == job_id:
            del self._active[workshop_id]
        record = self._records[job_id]
        if record.status not in TERMINAL_STATUSES:
            logger.error("Job {} ended without a terminal status", job_id)
            self.update(
                job_id,
                JobUpdateSchema(status=JobStatus.FAILED, error_code="internal_error"),
            )
