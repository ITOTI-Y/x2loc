from __future__ import annotations

import asyncio
import csv
import io
import math
import time
from typing import Final, Literal, Self

from httpx2 import (
    AsyncBaseTransport,
    AsyncClient,
    Limits,
    Request,
    Response,
    TransportError,
)
from loguru import logger

from src.models.weblate import (
    UNIT_PAGE_SIZE,
    CorpusUnitSchema,
    WeblateComponentDraftSchema,
    WeblateComponentSchema,
    WeblateConfigSchema,
    WeblatePageSchema,
    WeblateRequestParamsSchema,
    WeblateRequestSchema,
    WeblateTaskSchema,
    WeblateUnitDraftSchema,
    WeblateUnitPatchSchema,
    WeblateUnitSchema,
    WeblateUploadResultSchema,
)

WEBLATE_STATE_TRANSLATED: Final[int] = 20

# Ordinary calls must fail fast; only file uploads legitimately run for
# minutes while Weblate imports them.
HTTP_TIMEOUT: Final[float] = 30.0
HTTP_UPLOAD_TIMEOUT: Final[float] = 300.0
REQUEST_CONCURRENCY: Final[int] = 16
PAGINATE_CONCURRENCY: Final[int] = 8
KEEPALIVE_EXPIRY: Final[float] = 300.0

RETRY_MAX_ATTEMPTS: Final[int] = 4
RETRY_BASE_DELAY: Final[float] = 2.0
LOCK_BUSY_BASE_DELAY: Final[float] = 30.0
TASK_POLL_INTERVAL: Final[float] = 2.0
TASK_POLL_TIMEOUT: Final[float] = 900.0

UPLOAD_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "context",
    "source",
    "target",
    "developer_comments",
)

# `method="add"` only appends source strings, so Weblate's per-batch work is
# linear in batch size. 5000 rows was verified to import within the 300s
# client timeout; larger batches start hitting server-side import timeouts.
SOURCE_BATCH_SIZE: Final[int] = 5000

# `method="translate"` is O(batch) *and* O(component_total): every row is
# matched against existing source units by context. On a 26K-unit component a
# 5000-row target batch spends 100+s in Weblate's import pipeline and trips
# Cloudflare's fixed 524 proxy timeout. 500 keeps each batch well under it.
TARGET_BATCH_SIZE: Final[int] = 500

# `method="translate"` returning accepted=0 for a CSV full of non-empty
# targets means Weblate's unit index has not caught up with the preceding
# bulk source upload. Retry with backoff rather than dropping translations.
ZERO_ACCEPTED_RETRIES: Final[int] = 3
ZERO_ACCEPTED_BACKOFF: Final[float] = 30.0


class WeblateAPIError(RuntimeError):
    """Weblate returned a status the client cannot recover from.

    Carries only the status code and a short operation label. Response
    bodies may echo tokens or full payloads and never enter the message.
    """

    def __init__(self, status_code: int, operation: str) -> None:
        self.status_code = status_code
        self.operation = operation
        super().__init__(f"Weblate operation failed: {operation} ({status_code})")


async def _on_request(request: Request) -> None:
    request.extensions["x2loc_started"] = time.monotonic()


async def _on_response(response: Response) -> None:
    started = response.request.extensions.get("x2loc_started")
    if started is not None:
        logger.debug(
            "Weblate {} {} -> {} in {:.2f}s",
            response.request.method,
            response.request.url.path,
            response.status_code,
            time.monotonic() - started,
        )


def units_to_csv(
    units: list[CorpusUnitSchema], *, content: Literal["source", "target"]
) -> bytes:
    """Serialize units into a Weblate language-specific CSV file.

    Weblate treats a CSV as a language-specific file: the `target` column
    carries the actual text for that file's language, and `source` is only
    used for matching. `content="source"` builds the source-language docfile
    used for component creation and `method="add"`; `content="target"` builds
    the translation upload. Rows whose chosen value is empty are skipped.

    Every field is quoted. Weblate sniffs the delimiter by character frequency
    on the first data row, and a short glossary term such as
    `(VIP Capture Only)` contains more spaces than commas, which makes Weblate
    pick SPACE and garble the whole file. `QUOTE_ALL` removes the ambiguity.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(UPLOAD_CSV_COLUMNS), quoting=csv.QUOTE_ALL
    )
    writer.writeheader()
    for unit in units:
        value = unit.source if content == "source" else unit.target
        if not value:
            continue
        writer.writerow(
            {
                "context": unit.context,
                "source": unit.source,
                "target": value,
                "developer_comments": unit.note,
            }
        )
    return buffer.getvalue().encode("utf-8")


def _batched(units: list[CorpusUnitSchema], size: int) -> list[list[CorpusUnitSchema]]:
    return [units[start : start + size] for start in range(0, len(units), size)]


class AsyncWeblateClient:
    def __init__(
        self,
        config: WeblateConfigSchema,
        transport: AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.base_url = config.url.rstrip("/") + "/"
        self._client = AsyncClient(
            base_url=self.base_url,
            transport=transport,
            headers={
                "Authorization": f"Token {config.token.get_secret_value()}",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=HTTP_TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            http2=True,
            limits=Limits(
                max_connections=REQUEST_CONCURRENCY,
                max_keepalive_connections=REQUEST_CONCURRENCY,
                keepalive_expiry=KEEPALIVE_EXPIRY,
            ),
            event_hooks={"request": [_on_request], "response": [_on_response]},
        )
        self._request_sem = asyncio.Semaphore(REQUEST_CONCURRENCY)
        self._paginate_sem = asyncio.Semaphore(PAGINATE_CONCURRENCY)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def get_component(self, slug: str) -> WeblateComponentSchema | None:
        response = await self._request(
            WeblateRequestSchema(
                method="GET",
                path=f"components/{self.config.project_slug}/{slug}/",
            ),
            expected_statuses=frozenset({404}),
        )
        if response.status_code == 404:
            return None
        return WeblateComponentSchema.model_validate(response.json())

    async def create_component(self, draft: WeblateComponentDraftSchema) -> None:
        """Create a VCS-less component from an uploaded source document.

        The multipart field for VCS-less creation is `docfile`, not `file`;
        Weblate answers with a background task that must finish before the
        component's units exist.

        `manage_units` and `edit_template` are applied via a post-create
        PATCH: Weblate silently drops them on the creation POST, and the
        pair is required together — without `edit_template`, every later
        `method="add"` upload and `create_unit` call fails with "Adding
        strings is disabled in the component configuration" even when
        `manage_units` is True (verified against Weblate 5.x, 2026-04 and
        2026-07).
        """
        response = await self._request(
            WeblateRequestSchema(
                method="POST",
                path=f"projects/{self.config.project_slug}/components/",
                data={
                    "name": draft.name,
                    "slug": draft.slug,
                    "file_format": "csv",
                    "source_language": self.config.source_language,
                    "license": self.config.license,
                    "license_url": self.config.license_url,
                },
                files={"docfile": (f"{draft.slug}.csv", draft.source_csv, "text/csv")},
            ),
            timeout=HTTP_UPLOAD_TIMEOUT,
        )
        await self._await_task(response.json().get("task_url"))
        await self._request(
            WeblateRequestSchema(
                method="PATCH",
                path=f"components/{self.config.project_slug}/{draft.slug}/",
                json_body={"manage_units": True, "edit_template": True},
            )
        )
        logger.success(f"Created component {draft.slug}")

    async def ensure_translation(self, component_slug: str, language: str) -> None:
        """Create the target translation, tolerating an existing one."""
        await self._request(
            WeblateRequestSchema(
                method="POST",
                path=(
                    f"components/{self.config.project_slug}/{component_slug}"
                    "/translations/"
                ),
                data={"language_code": language},
            ),
            expected_statuses=frozenset({400, 409}),
        )

    async def upload_file(
        self,
        component_slug: str,
        language: str,
        payload: bytes,
        *,
        method: Literal["add", "translate"],
        conflicts: str = "",
    ) -> WeblateUploadResultSchema:
        data = {"method": method} | ({"conflicts": conflicts} if conflicts else {})
        response = await self._request(
            WeblateRequestSchema(
                method="POST",
                path=(
                    f"translations/{self.config.project_slug}/{component_slug}"
                    f"/{language}/file/"
                ),
                data=data,
                files={"file": (f"{component_slug}.csv", payload, "text/csv")},
            ),
            timeout=HTTP_UPLOAD_TIMEOUT,
        )
        result = WeblateUploadResultSchema.model_validate(response.json())
        await self._await_task(result.task_url)
        return result

    async def list_units_page(
        self,
        component_slug: str,
        language: str,
        params: WeblateRequestParamsSchema,
    ) -> WeblatePageSchema[WeblateUnitSchema]:
        response = await self._request(
            WeblateRequestSchema(
                method="GET",
                path=(
                    f"translations/{self.config.project_slug}/{component_slug}"
                    f"/{language}/units/"
                ),
                params=params,
            )
        )
        return WeblatePageSchema[WeblateUnitSchema].model_validate(response.json())

    async def list_units(
        self, component_slug: str, language: str, q: str = ""
    ) -> list[WeblateUnitSchema]:
        first = await self.list_units_page(
            component_slug, language, WeblateRequestParamsSchema(q=q)
        )
        total_pages = math.ceil(first.count / UNIT_PAGE_SIZE)
        if total_pages <= 1:
            return list(first.results)

        async def fetch(page: int) -> list[WeblateUnitSchema]:
            async with self._paginate_sem:
                result = await self.list_units_page(
                    component_slug,
                    language,
                    WeblateRequestParamsSchema(page=page, q=q),
                )
            return list(result.results)

        rest = await asyncio.gather(
            *(fetch(page) for page in range(2, total_pages + 1))
        )
        return [*first.results, *(unit for page in rest for unit in page)]

    async def search_units(
        self, params: WeblateRequestParamsSchema
    ) -> list[WeblateUnitSchema]:
        """Search across the whole instance; returns the first page only.

        Every consumer caps how many results it keeps, so paginating the
        whole result set only wastes round trips.
        """
        response = await self._request(
            WeblateRequestSchema(method="GET", path="units/", params=params)
        )
        page = WeblatePageSchema[WeblateUnitSchema].model_validate(response.json())
        return list(page.results)

    async def create_unit(
        self,
        component_slug: str,
        draft: WeblateUnitDraftSchema,
    ) -> None:
        """Create one source string on a template component.

        Weblate rejects unit creation on non-source translations with "Add
        the string to the source language instead"; the source-side body is
        the monolingual key/value shape with plural-form list value
        (verified against Weblate 5.x, 2026-07). Targets are filled
        afterwards via `upload_targets`.
        """
        await self._request(
            WeblateRequestSchema(
                method="POST",
                path=(
                    f"translations/{self.config.project_slug}/{component_slug}"
                    f"/{self.config.source_language}/units/"
                ),
                json_body={"key": draft.context, "value": [draft.source]},
            )
        )

    async def upload_targets(
        self, component_slug: str, language: str, units: list[CorpusUnitSchema]
    ) -> None:
        """Fill existing units' targets in batches with catch-up retries."""
        for batch in _batched(units, TARGET_BATCH_SIZE):
            payload = units_to_csv(batch, content="target")
            if payload.count(b"\n") > 1:
                await self._upload_targets(component_slug, language, payload)

    async def patch_unit(self, unit_id: int, patch: WeblateUnitPatchSchema) -> None:
        """Weblate answers a unit PATCH with a partial body (no id/source/
        context on 5.x); nothing consumes it, so it is not parsed."""
        await self._request(
            WeblateRequestSchema(
                method="PATCH",
                path=f"units/{unit_id}/",
                json_body=patch.model_dump(),
            )
        )

    async def sync_corpus(
        self,
        component_slug: str,
        *,
        name: str,
        language: str,
        units: list[CorpusUnitSchema],
        has_existing_target: bool,
    ) -> None:
        """Bring one component's source strings and existing targets up to date.

        Creates the component from a source docfile when it is missing,
        otherwise appends new source strings with `method="add"`. Existing
        targets are filled with `method="translate", conflicts="ignore"` so
        Weblate keeps whatever it already holds — Weblate is the authority.

        An empty unit list is a caller error: creating a header-only
        component is undefined behavior on Weblate's side, and the pipeline
        filters untranslatable files before syncing.
        """
        if not units:
            raise ValueError(f"sync_corpus received no units for {component_slug}")
        source_batches = _batched(units, SOURCE_BATCH_SIZE)
        if await self.get_component(component_slug) is None:
            await self.create_component(
                WeblateComponentDraftSchema(
                    name=name,
                    slug=component_slug,
                    source_csv=units_to_csv(source_batches[0], content="source"),
                )
            )
            source_batches = source_batches[1:]
        for batch in source_batches:
            await self.upload_file(
                component_slug,
                self.config.source_language,
                units_to_csv(batch, content="source"),
                method="add",
            )

        await self.ensure_translation(component_slug, language)
        if not has_existing_target:
            return
        await self.upload_targets(component_slug, language, units)

    async def _upload_targets(
        self, component_slug: str, language: str, payload: bytes
    ) -> None:
        for attempt in range(1, ZERO_ACCEPTED_RETRIES + 1):
            result = await self.upload_file(
                component_slug,
                language,
                payload,
                method="translate",
                conflicts="ignore",
            )
            if result.accepted or result.skipped or result.not_found:
                return
            if attempt < ZERO_ACCEPTED_RETRIES:
                logger.warning(
                    "Weblate accepted 0 translations for {}; unit index is still "
                    "catching up, retry {}/{} after {:.0f}s",
                    component_slug,
                    attempt,
                    ZERO_ACCEPTED_RETRIES,
                    ZERO_ACCEPTED_BACKOFF,
                )
                await asyncio.sleep(ZERO_ACCEPTED_BACKOFF)
        raise WeblateAPIError(502, f"translate upload to {component_slug}")

    async def _await_task(self, task_url: str | None) -> None:
        if not task_url:
            return
        deadline = time.monotonic() + TASK_POLL_TIMEOUT
        while time.monotonic() < deadline:
            response = await self._request(
                WeblateRequestSchema(method="GET", path=task_url)
            )
            task = WeblateTaskSchema.model_validate(response.json())
            if task.completed:
                if task.result and task.result.error:
                    raise WeblateAPIError(502, "weblate background task")
                return
            await asyncio.sleep(TASK_POLL_INTERVAL)
        raise WeblateAPIError(504, "weblate background task")

    async def _request(
        self,
        request: WeblateRequestSchema,
        *,
        timeout: float = HTTP_TIMEOUT,
        expected_statuses: frozenset[int] = frozenset(),
    ) -> Response:
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                async with self._request_sem:
                    response = await self._client.request(
                        request.method,
                        request.path,
                        params=request.query_params(),
                        json=request.json_body,
                        data=request.data,
                        files=request.files,
                        timeout=timeout,
                    )
            except TransportError as exc:
                if attempt == RETRY_MAX_ATTEMPTS:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Weblate network error on {} {}: {!r}; retry {}/{} after {:.0f}s",
                    request.method,
                    request.path,
                    exc,
                    attempt,
                    RETRY_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if response.status_code < 400 or (
                response.status_code in expected_statuses
            ):
                return response
            delay = self._retry_delay(response, attempt)
            if delay is None or attempt == RETRY_MAX_ATTEMPTS:
                raise WeblateAPIError(
                    response.status_code, f"{request.method} {request.path}"
                )
            logger.warning(
                "Weblate {} on {} {}; retry {}/{} after {:.0f}s",
                response.status_code,
                request.method,
                request.path,
                attempt,
                RETRY_MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)
        raise AssertionError("unreachable: every attempt returns or raises")

    @staticmethod
    def _retry_delay(response: Response, attempt: int) -> float | None:
        """Seconds to wait before retrying, or None when not retryable.

        Four server conditions share one recovery strategy — a long wait, then
        retry — because the origin may still be completing the request:

          - **504 Gateway Timeout** — the proxy gave up while Weblate imported.
          - **524** — Cloudflare's equivalent, fired when origin processing
            exceeds its fixed upload window.
          - **423 Locked** — Weblate's repository lock is held by a background
            task, e.g. unit materialization right after `create_component`.
          - **400 containing "could not be acquired"** — the `component-update`
            lock is held by a still-running request.
        """
        status = response.status_code
        if status in (423, 504, 524) or (
            status == 400 and b"could not be acquired" in response.content
        ):
            return LOCK_BUSY_BASE_DELAY * (2 ** (attempt - 1))
        if status == 429 or 500 <= status < 600:
            return RETRY_BASE_DELAY * (2 ** (attempt - 1))
        return None
