from __future__ import annotations

import hmac
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from httpx2 import AsyncClient
from sse_starlette import EventSourceResponse

from src.api.config import ServiceConfigSchema
from src.core.workshop import WorkshopInputError
from src.jobs._share import SSE_PING_SECONDS
from src.jobs.manager import JobManager
from src.jobs.pipeline import WorkshopPipeline
from src.models.job import (
    JobCreateResponseSchema,
    JobRecordSchema,
    JobStatus,
    JobStatusResponseSchema,
    WorkshopJobRequestSchema,
)
from src.services.steam import SteamDownloadError, fetch_workshop_metadata


@dataclass(frozen=True)
class ServiceResources:
    manager: JobManager
    pipeline: WorkshopPipeline
    steam_web: AsyncClient


ResourceFactory = Callable[[], AbstractAsyncContextManager[ServiceResources]]


def job_response(job: JobRecordSchema) -> JobStatusResponseSchema:
    ready = job.status == JobStatus.SUCCEEDED and job.artifact is not None
    return JobStatusResponseSchema(
        id=job.id,
        workshop_id=job.workshop_id,
        title=job.title,
        target_lang=job.target_lang,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        error_code=job.error_code,
        artifact_sha256=job.artifact.sha256 if job.artifact else None,
        artifact_bytes=job.artifact.bytes if job.artifact else None,
        artifact_url=f"/v1/jobs/{job.id}/artifact" if ready else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
    )


def create_app(
    config: ServiceConfigSchema,
    resource_factory: ResourceFactory,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with resource_factory() as resources:
            app.state.resources = resources
            try:
                yield
            finally:
                await resources.manager.close()

    app = FastAPI(title="x2loc Workshop API", version="1.0.0", lifespan=lifespan)

    async def authorize(authorization: Annotated[str, Header()] = "") -> None:
        expected = f"Bearer {config.service_token.get_secret_value()}"
        if not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    def resources(request: Request) -> ServiceResources:
        return request.app.state.resources

    Resources = Annotated[ServiceResources, Depends(resources)]

    @app.post(
        "/v1/jobs",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authorize)],
    )
    async def create_job(
        body: WorkshopJobRequestSchema, current: Resources
    ) -> JobCreateResponseSchema:
        try:
            metadata = await fetch_workshop_metadata(
                body.workshop_id, client=current.steam_web
            )
        except WorkshopInputError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except SteamDownloadError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
            ) from exc
        job = current.manager.submit(
            body, title=metadata.title, runner=current.pipeline.run
        )
        return JobCreateResponseSchema(
            id=job.id,
            status=job.status,
            status_url=f"/v1/jobs/{job.id}",
            events_url=f"/v1/jobs/{job.id}/events",
        )

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    async def get_job(job_id: str, current: Resources) -> JobStatusResponseSchema:
        job = current.manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return job_response(job)

    @app.get("/v1/jobs/{job_id}/events", dependencies=[Depends(authorize)])
    async def stream_job(job_id: str, current: Resources) -> EventSourceResponse:
        if current.manager.get(job_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        async def stream() -> AsyncIterator[dict[str, str]]:
            async for record in current.manager.events(job_id):
                yield {
                    "event": "status",
                    "data": job_response(record).model_dump_json(),
                }

        return EventSourceResponse(stream(), ping=SSE_PING_SECONDS)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    async def cancel_job(job_id: str, current: Resources) -> JobStatusResponseSchema:
        try:
            job = current.manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        return job_response(job)

    @app.get("/v1/jobs/{job_id}/artifact", dependencies=[Depends(authorize)])
    async def download_artifact(job_id: str, current: Resources) -> FileResponse:
        job = current.manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if job.status != JobStatus.SUCCEEDED or job.artifact is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT)
        path = config.artifact_root / job.artifact.path.name
        if not path.is_file():
            raise HTTPException(status_code=status.HTTP_410_GONE)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=path.name,
            headers={"X-Artifact-SHA256": job.artifact.sha256},
        )

    return app
