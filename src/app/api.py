"""HTTP interface of the local app, served on loopback to its own pages.

Every `/api` request carries the session token the launcher generated, so
other local programs and web pages cannot drive the app; the `Host` check
stops a foreign site from reaching it through DNS rebinding.
"""

import asyncio
import contextlib
import hmac
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final, Literal

import httpx
from fastapi import Body, FastAPI, Request, Response
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from httpx2 import TransportError
from loguru import logger
from pydantic import ConfigDict, Field
from sse_starlette import EventSourceResponse

from src.agent.transport import build_llm_http_client
from src.api.app import job_response
from src.app.catalog import Coverage, ModEntry, ModProblem, coverage, scan_mods
from src.app.overlays import (
    InstalledOverlay,
    list_overlays,
    uninstall_overlay,
)
from src.app.runs import Run, RunError, RunManager, RunStatus
from src.app.secrets import SecretName, Secrets
from src.app.settings import AppSettingsSchema, load_settings, save_settings
from src.app.steam import detect_paths
from src.config import LocalConfigSchema, ServiceConfigSchema
from src.core.overlay_mod import overlay_mod_name
from src.jobs._share import SSE_PING_SECONDS
from src.models._share import BaseSchema
from src.models.job import JobStatusResponseSchema
from src.models.store import StoreMode
from src.models.weblate import (
    WeblateAccount,
    WeblateTranslationStatsSchema,
    WeblateUnitPatchSchema,
)
from src.models.workshop import TARGET_LANGUAGE
from src.services.stores import LocalStore, QueuedTranslation, UploadReport
from src.services.stores.local import Origin
from src.services.stores.upstream import component_of, needs_editing_units
from src.services.weblate import (
    WEBLATE_STATE_TRANSLATED,
    AsyncWeblateClient,
    WeblateAPIError,
)

LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "localhost"})
SESSION_HEADER: Final = "X-Session-Token"
WORKSHOP_ID: Final = r"^\d+$"


class AppError(Exception):
    """A request the app refuses; `code` is what the pages translate."""

    def __init__(self, code: str, status: int, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.detail = detail


class SettingsViewSchema(BaseSchema):
    settings: AppSettingsSchema
    # Whether each secret is set; their values never leave the app.
    secrets: dict[SecretName, bool]


class SecretValueSchema(BaseSchema):
    value: str = Field(min_length=1)


class CheckSchema(BaseSchema):
    ok: bool
    code: str | None = None
    detail: str = ""


class WeblateCheckSchema(CheckSchema):
    username: str | None = None
    can_write: bool = False


class SettingsCheckSchema(BaseSchema):
    llm: CheckSchema
    weblate: WeblateCheckSchema


class DetectedPathsSchema(BaseSchema):
    workshop_dir: Path | None
    mods_dir: Path | None


class ModViewSchema(BaseSchema):
    workshop_id: str
    title: str
    slugs: list[str]
    overlay_installed: bool
    problem: ModProblem | None
    coverage: Coverage | None


class ModsViewSchema(BaseSchema):
    mods: list[ModViewSchema]
    # Weblate's counts arrive in the background; until then coverage is None.
    coverage_state: Literal["loading", "ready", "failed"]


class RunRequestSchema(BaseSchema):
    workshop_ids: list[Annotated[str, Field(pattern=WORKSHOP_ID)]] = Field(min_length=1)
    upload_after: bool = False


class RunViewSchema(BaseSchema):
    id: str
    status: RunStatus
    workshop_ids: list[str]
    upload_after: bool
    started_at: datetime
    finished_at: datetime | None
    skipped: dict[str, str]
    jobs: dict[str, JobStatusResponseSchema]
    installed: dict[str, str]
    upload: UploadReport | None
    error_code: str | None


class OverlaysViewSchema(BaseSchema):
    overlays: list[InstalledOverlay]


class TextSchema(BaseSchema):
    """Carries translations, whose surrounding whitespace is part of them."""

    model_config = BaseSchema.model_config | ConfigDict(str_strip_whitespace=False)


class ReviewItemSchema(TextSchema):
    where: Literal["weblate", "local"]
    slug: str
    context: str
    source: str
    target: str
    # Weblate's translation as read, which an edit kept here records.
    upstream: str
    comment: str
    # Weblate's unit, when the item came from Weblate.
    unit_id: int | None = None
    web_url: str = ""


class ReviewEditSchema(TextSchema):
    slug: str
    context: str
    source: str
    target: str = Field(min_length=1)
    upstream: str
    unit_id: int | None = None


class UploadRequestSchema(BaseSchema):
    # Components to upload; None uploads the whole queue.
    slugs: list[str] | None = None


class ResolveRequestSchema(TextSchema):
    slug: str
    context: str
    origin: Origin
    keep: Literal["local", "upstream"]


@dataclass
class _Coverage:
    stats: dict[str, WeblateTranslationStatsSchema] | None = None
    task: asyncio.Task[None] | None = None
    failed: bool = False


@dataclass
class AppState:
    home: Path
    secrets: Secrets
    runs: RunManager = field(default_factory=RunManager)
    mods: tuple[Path, list[ModEntry]] | None = None
    coverage: _Coverage = field(default_factory=_Coverage)
    # Account of the last token checked, keyed by the token itself.
    account: tuple[str, WeblateAccount] | None = None

    def settings(self) -> AppSettingsSchema:
        return load_settings(self.home)

    async def config(self) -> ServiceConfigSchema:
        """The pipeline config of the saved settings and secrets.

        In `auto`, a token that cannot write is used to read only, so its
        jobs keep their translations here instead of failing on Weblate.
        """
        settings = self.settings()
        config = settings.to_service_config(
            home=self.home,
            llm_api_key=self.secrets.get("llm_api_key"),
            weblate_token=self.secrets.get("weblate_token"),
        )
        if config.store == StoreMode.AUTO and config.writes_weblate:
            account = await self.weblate_account(config)
            if not account.can_write:
                return config.model_copy(update={"store": StoreMode.LOCAL})
        return config

    async def weblate_account(self, config: ServiceConfigSchema) -> WeblateAccount:
        token = config.weblate.token.get_secret_value()
        if self.account is not None and self.account[0] == token:
            return self.account[1]
        async with AsyncWeblateClient(config.weblate) as client:
            account = await client.account()
        self.account = (token, account)
        return account


def paths_of(config: ServiceConfigSchema) -> LocalConfigSchema:
    if config.local is None:
        raise AppError("paths_not_set", 409)
    return config.local


@asynccontextmanager
async def local_store(
    config: ServiceConfigSchema,
) -> AsyncGenerator[tuple[AsyncWeblateClient, LocalStore]]:
    async with AsyncWeblateClient(config.weblate) as client:
        store = LocalStore(client, config.store_path)
        try:
            yield client, store
        finally:
            await store.aclose()


async def check_llm(settings: AppSettingsSchema, api_key: str) -> CheckSchema:
    """Ask the translation model for one token."""
    model = settings.model
    if not model.translation_model:
        return CheckSchema(ok=False, code="model_not_set")
    if not api_key:
        return CheckSchema(ok=False, code="api_key_not_set")
    headers = {"Authorization": f"Bearer {api_key}"}
    async with build_llm_http_client(model.timeout_seconds) as http:
        try:
            response = await http.post(
                model.base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json={
                    "model": model.translation_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
        except httpx.HTTPError as exc:
            return CheckSchema(
                ok=False, code="llm_unreachable", detail=type(exc).__name__
            )
    if response.status_code in (401, 403):
        return CheckSchema(ok=False, code="llm_unauthorized")
    if response.status_code >= 400:
        return CheckSchema(
            ok=False, code="llm_rejected", detail=str(response.status_code)
        )
    return CheckSchema(ok=True)


async def check_weblate(
    state: AppState, config: ServiceConfigSchema
) -> WeblateCheckSchema:
    state.account = None
    try:
        account = await state.weblate_account(config)
    except WeblateAPIError as exc:
        code = "token_invalid" if exc.status_code in (401, 403) else "weblate_rejected"
        return WeblateCheckSchema(ok=False, code=code, detail=str(exc.status_code))
    except TransportError as exc:
        return WeblateCheckSchema(
            ok=False, code="weblate_unreachable", detail=type(exc).__name__
        )
    return WeblateCheckSchema(
        ok=True, username=account.username, can_write=account.can_write
    )


def run_view(runs: RunManager, run: Run) -> RunViewSchema:
    jobs = {
        workshop_id: job_response(record)
        for workshop_id, job_id in run.jobs.items()
        if (record := runs.job(job_id)) is not None
    }
    return RunViewSchema(
        id=run.id,
        status=run.status,
        workshop_ids=run.workshop_ids,
        upload_after=run.upload_after,
        started_at=run.started_at,
        finished_at=run.finished_at,
        skipped=run.skipped,
        jobs=jobs,
        installed=run.installed,
        upload=run.upload,
        error_code=run.error_code,
    )


async def _fetch_coverage(state: AppState, config: ServiceConfigSchema) -> None:
    try:
        async with AsyncWeblateClient(config.weblate) as client:
            state.coverage.stats = await client.project_translation_stats(
                TARGET_LANGUAGE
            )
    except (WeblateAPIError, TransportError):
        logger.exception("Reading translation counts from Weblate failed")
        state.coverage.failed = True


def create_app(*, home: Path, session_token: str, secrets: Secrets) -> FastAPI:
    state = AppState(home=home, secrets=secrets)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await state.runs.aclose()
            task = state.coverage.task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="x2loc", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        host = request.headers.get("host", "").rsplit(":", 1)[0]
        if host not in LOOPBACK_HOSTS:
            return JSONResponse({"code": "forbidden_host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            # EventSource cannot set headers, so streams pass it as a query.
            given = request.headers.get(SESSION_HEADER) or request.query_params.get(
                "token", ""
            )
            if not hmac.compare_digest(given.encode(), session_token.encode()):
                return JSONResponse({"code": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(AppError)
    async def app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            {"code": exc.code, "detail": exc.detail}, status_code=exc.status
        )

    @app.exception_handler(WeblateAPIError)
    async def weblate_error(_request: Request, exc: WeblateAPIError) -> JSONResponse:
        code = (
            "weblate_forbidden" if exc.status_code in (401, 403) else "weblate_rejected"
        )
        return JSONResponse(
            {"code": code, "detail": str(exc.status_code)}, status_code=502
        )

    @app.exception_handler(TransportError)
    async def transport_error(_request: Request, exc: TransportError) -> JSONResponse:
        return JSONResponse(
            {"code": "weblate_unreachable", "detail": type(exc).__name__},
            status_code=502,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(p) for p in error["loc"]) for error in exc.errors()]
        return JSONResponse(
            {"code": "invalid_request", "detail": ", ".join(fields)}, status_code=422
        )

    @app.get("/api/settings")
    async def get_settings() -> SettingsViewSchema:
        return SettingsViewSchema(
            settings=state.settings(), secrets=state.secrets.presence()
        )

    @app.put("/api/settings")
    async def put_settings(settings: AppSettingsSchema) -> SettingsViewSchema:
        save_settings(state.home, settings)
        state.mods = None
        return SettingsViewSchema(settings=settings, secrets=state.secrets.presence())

    @app.put("/api/secrets/{name}", status_code=204)
    async def put_secret(name: SecretName, body: SecretValueSchema) -> None:
        state.secrets.set(name, body.value)
        state.account = None

    @app.delete("/api/secrets/{name}", status_code=204)
    async def delete_secret(name: SecretName) -> None:
        state.secrets.delete(name)
        state.account = None

    @app.post("/api/settings/test")
    async def test_settings() -> SettingsCheckSchema:
        settings = state.settings()
        config = settings.to_service_config(
            home=state.home,
            llm_api_key=state.secrets.get("llm_api_key"),
            weblate_token=state.secrets.get("weblate_token"),
        )
        llm, weblate = await asyncio.gather(
            check_llm(settings, state.secrets.get("llm_api_key")),
            check_weblate(state, config),
        )
        return SettingsCheckSchema(llm=llm, weblate=weblate)

    @app.get("/api/paths/detect")
    async def detect() -> DetectedPathsSchema:
        found = await asyncio.to_thread(detect_paths)
        return DetectedPathsSchema(
            workshop_dir=found.workshop_dir, mods_dir=found.mods_dir
        )

    @app.get("/api/mods")
    async def get_mods(refresh: bool = False) -> ModsViewSchema:
        config = await state.config()
        local = paths_of(config)
        if refresh or state.mods is None or state.mods[0] != local.workshop_dir:
            entries = await asyncio.to_thread(
                scan_mods, local.workshop_dir, local.mods_dir, config.limits
            )
            state.mods = (local.workshop_dir, entries)
        held = state.coverage
        if refresh or (held.task is None or (held.failed and held.task.done())):
            held.stats, held.failed = None, False
            held.task = asyncio.create_task(_fetch_coverage(state, config))
        stats = held.stats
        mods = [
            ModViewSchema(
                workshop_id=entry.workshop_id,
                title=entry.title,
                slugs=list(entry.slugs),
                # Installs and uninstalls happen between scans.
                overlay_installed=(
                    local.mods_dir / overlay_mod_name(entry.workshop_id)
                ).is_dir(),
                problem=entry.problem,
                coverage=coverage(entry, stats) if stats is not None else None,
            )
            for entry in state.mods[1]
        ]
        coverage_state = (
            "failed"
            if held.failed
            else "ready"
            if held.stats is not None
            else "loading"
        )
        return ModsViewSchema(mods=mods, coverage_state=coverage_state)

    @app.post("/api/runs", status_code=202)
    async def start_run(body: RunRequestSchema) -> RunViewSchema:
        settings = state.settings()
        # No model or API key is checked here: mods Weblate has translated in
        # full need neither, and a mod that does fails with
        # `llm_not_configured`.
        config = await state.config()
        try:
            run = state.runs.start(
                config,
                body.workshop_ids,
                jobs=settings.jobs,
                llm_concurrency=settings.model.concurrency,
                upload_after=body.upload_after,
            )
        except RunError as exc:
            raise AppError(exc.code, 409) from exc
        return run_view(state.runs, run)

    @app.get("/api/runs")
    async def list_runs() -> list[RunViewSchema]:
        return [run_view(state.runs, run) for run in state.runs.runs()]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> RunViewSchema:
        run = state.runs.get(run_id)
        if run is None:
            raise AppError("run_not_found", 404)
        return run_view(state.runs, run)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> RunViewSchema:
        try:
            run = state.runs.cancel(run_id)
        except KeyError as exc:
            raise AppError("run_not_found", 404) from exc
        return run_view(state.runs, run)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> JobStatusResponseSchema:
        record = state.runs.job(job_id)
        if record is None:
            raise AppError("job_not_found", 404)
        return job_response(record)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> EventSourceResponse:
        if state.runs.job(job_id) is None:
            raise AppError("job_not_found", 404)

        async def stream() -> AsyncIterator[dict[str, str]]:
            async for record in state.runs.events(job_id):
                yield {
                    "event": "status",
                    "data": job_response(record).model_dump_json(),
                }

        return EventSourceResponse(stream(), ping=SSE_PING_SECONDS)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> JobStatusResponseSchema:
        try:
            record = state.runs.cancel_job(job_id)
        except KeyError as exc:
            raise AppError("job_not_found", 404) from exc
        return job_response(record)

    @app.get("/api/overlays")
    async def get_overlays() -> OverlaysViewSchema:
        mods_dir = paths_of(await state.config()).mods_dir
        overlays = await asyncio.to_thread(list_overlays, mods_dir)
        return OverlaysViewSchema(overlays=overlays)

    @app.delete("/api/overlays/{workshop_id}", status_code=204)
    async def delete_overlay(
        workshop_id: Annotated[str, PathParam(pattern=WORKSHOP_ID)],
    ) -> None:
        mods_dir = paths_of(await state.config()).mods_dir
        try:
            await asyncio.to_thread(uninstall_overlay, mods_dir, workshop_id)
        except FileNotFoundError as exc:
            raise AppError("overlay_not_found", 404) from exc

    @app.get("/api/review")
    async def get_review() -> list[ReviewItemSchema]:
        config = await state.config()
        async with local_store(config) as (client, store):
            units = await needs_editing_units(client, TARGET_LANGUAGE)
            local = store.review(TARGET_LANGUAGE)
            # Units a player already corrected here wait in the upload queue.
            edited = {
                (row.slug, row.context)
                for row in store.queue(uploads=("pending", "uploaded", "conflict"))
                if row.origin == "manual"
            }
        items = [
            ReviewItemSchema(
                where="weblate",
                slug=component_of(unit),
                context=unit.context,
                source=unit.source,
                target=unit.target,
                upstream=unit.target,
                comment="",
                unit_id=unit.id,
                web_url=unit.web_url,
            )
            for unit in units
        ]
        items = [i for i in items if (i.slug, i.context) not in edited]
        items += [
            ReviewItemSchema(
                where="local",
                slug=row.slug,
                context=row.context,
                source=row.source,
                target=row.target,
                upstream=row.upstream_at_read,
                comment=row.comment,
            )
            for row in local
        ]
        return items

    @app.put("/api/review", status_code=204)
    async def put_review(body: ReviewEditSchema) -> None:
        config = await state.config()
        if config.writes_weblate:
            if body.unit_id is None:
                raise AppError("unit_not_on_weblate", 409)
            async with AsyncWeblateClient(config.weblate) as client:
                await client.patch_unit(
                    body.unit_id,
                    WeblateUnitPatchSchema(
                        target=[body.target], state=WEBLATE_STATE_TRANSLATED
                    ),
                )
            return
        async with local_store(config) as (_client, store):
            store.edit(
                body.slug,
                TARGET_LANGUAGE,
                body.context,
                source=body.source,
                target=body.target,
                upstream_at_read=body.upstream,
            )

    @app.get("/api/uploads")
    async def get_uploads() -> list[QueuedTranslation]:
        async with local_store(await state.config()) as (_client, store):
            return store.queue()

    @app.post("/api/uploads")
    async def post_uploads(
        body: Annotated[UploadRequestSchema, Body()],
    ) -> UploadReport:
        config = await state.config()
        if not config.weblate.token.get_secret_value():
            raise AppError("token_not_set", 409)
        if not (await state.weblate_account(config)).can_write:
            raise AppError("token_read_only", 409)
        slugs = set(body.slugs) if body.slugs is not None else None
        async with local_store(config) as (client, store):
            return await store.upload(client, language=TARGET_LANGUAGE, slugs=slugs)

    @app.post("/api/uploads/resolve", status_code=204)
    async def resolve_upload(body: ResolveRequestSchema) -> None:
        config = await state.config()
        async with local_store(config) as (_client, store):
            store.resolve(
                body.slug, TARGET_LANGUAGE, body.context, body.origin, keep=body.keep
            )

    return app
