import asyncio
import math
import time
import tomllib
from pathlib import Path
from typing import Any, Final, Self, cast

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
    WeblateComponentSchema,
    WeblateConfigSchema,
    WeblatePageSchema,
    WeblateRequestParamsSchema,
    WeblateRequestSchema,
    WeblateUnitSchema,
)


def load_weblate_config(path: Path) -> WeblateConfigSchema:
    """Load a flat Weblate TOML config (url/token/project_slug at top level)."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    return WeblateConfigSchema(
        url=data["url"],
        token=data["token"],
        project_slug=data["project_slug"],
        license=data.get("license", ""),
        license_url=data.get("license_url", ""),
    )


RATE_LIMIT_FLOOR: Final[int] = 100


HTTP_TIMEOUT: Final[float] = 60.0
HTTP_UPLOAD_TIMEOUT: Final[float] = 300.0
LOCK_BUSY_BASE_DELAY: Final[float] = 60.0
LOCK_BUSY_ERROR_SUBSTRING: Final[str] = "could not be acquired"
TASK_POLL_INTERVAL: Final[float] = 2.0
TASK_POLL_TIMEOUT: Final[float] = 300.0

# async settings
RETRY_MAX_ATTEMPTS: Final[int] = 3
RETRY_BASE_DELAY: Final[float] = 1.0
PAGINATE_CONCURRENCY: Final[int] = 10
REQUEST_CONCURRENCY: Final[int] = 16
TRANSLATION_CACHE_TTL: Final[float] = 300.0
KEEPALIVE_EXPIRY: Final[float] = 300.0


class WeblateAPIError(Exception):
    """Raised when the Weblate API returns an unrecoverable error."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


async def _on_request(request: Request) -> None:
    request.extensions["t0"] = time.monotonic()


async def _on_response(response: Response) -> None:
    t0 = response.request.extensions.get("t0")
    if t0 is not None:
        url = response.request.url
        q = f"?{url.query.decode()}" if url.query else ""
        logger.debug(
            f"weblate {response.request.method} {url.path}{q[:120]} "
            f"-> {response.status_code} in {time.monotonic() - t0:.2f}s"
        )


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
                "Authorization": f"Token {config.token}",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            http2=True,
            limits=Limits(
                max_connections=REQUEST_CONCURRENCY,
                max_keepalive_connections=REQUEST_CONCURRENCY,
                keepalive_expiry=KEEPALIVE_EXPIRY,
            ),
            event_hooks={
                "request": [_on_request],
                "response": [_on_response],
            },
        )
        self._request_sem = asyncio.Semaphore(REQUEST_CONCURRENCY)
        self._paginate_sem = asyncio.Semaphore(PAGINATE_CONCURRENCY)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def get_project(self) -> dict[str, Any]:
        r = await self._request(
            WeblateRequestSchema(
                method="GET", path=f"projects/{self.config.project_slug}/"
            )
        )
        return r.json()

    async def create_component(
        self,
        *,
        component: WeblateComponentSchema,
    ) -> None:
        try:
            await self._request(
                WeblateRequestSchema(
                    method="POST",
                    path=f"projects/{self.config.project_slug}/components/",
                    data={
                        "name": component.name,
                        "slug": component.slug,
                        "file_format": "csv",
                        "source_language": component.source_language.code,
                        "source_csv": component.source_csv,
                        "license": self.config.license,
                        "license_url": self.config.license_url,
                    },
                    files={
                        "docfile": (
                            f"{component.slug}.csv",
                            component.source_csv,
                            "text/csv",
                        )
                    },
                ),
                timeout=HTTP_UPLOAD_TIMEOUT,
            )
            logger.success(f"Created component {component.slug}")
        except WeblateAPIError as exc:
            if exc.status == 400:
                logger.warning(
                    f"Failed to create component {component.slug}: {exc.message}"
                )
            else:
                raise

    async def delete_component(self, component_slug: str) -> None:
        try:
            r = await self._request(
                WeblateRequestSchema(
                    method="DELETE",
                    path=f"components/{self.config.project_slug}/{component_slug}/",
                )
            )
            if r.status_code == 204:
                logger.success(f"Deleted component {component_slug}")
        except WeblateAPIError as exc:
            if exc.status in {404}:
                logger.success(f"Deleted component {component_slug}")
            else:
                raise

    async def list_components(self):
        return await self._paginate(
            WeblateRequestSchema(
                method="GET",
                path=f"projects/{self.config.project_slug}/components/",
            ),
            WeblateComponentSchema,
        )

    async def list_units_page(
        self,
        component_slug: str,
        lang: str,
        page: int,
        page_size: int,
        q: str | None = None,
    ) -> WeblatePageSchema:
        path = f"translations/{self.config.project_slug}/{component_slug}/{lang}/units/"
        params = WeblateRequestParamsSchema(page=page, page_size=page_size, q=q)
        r = await self._request(
            WeblateRequestSchema(
                method="GET",
                path=path,
                params=params,
            )
        )
        return WeblatePageSchema.model_validate(r.json())

    async def search_units(
        self, params: WeblateRequestParamsSchema
    ) -> list[WeblateUnitSchema]:
        r = await self._request(
            WeblateRequestSchema(
                method="GET",
                path="units/",
                params=params,
            )
        )
        return [
            WeblateUnitSchema.model_validate(unit)
            for unit in r.json().get("results", [])
        ]

    async def list_units(
        self,
        component_slug: str,
        lang: str,
        q: str | None = None,
    ) -> list[WeblateUnitSchema]:
        units = await self._paginate(
            WeblateRequestSchema(
                method="GET",
                path=f"translations/{self.config.project_slug}/{component_slug}/{lang}/units/",
                params=WeblateRequestParamsSchema(q=q),
            ),
            WeblateUnitSchema,
        )
        return units

    async def patch_unit(self, unit_id: int, data: dict[str, Any]) -> dict[str, Any]:
        r = await self._request(
            WeblateRequestSchema(
                method="PATCH",
                path=f"units/{unit_id}/",
                json_body=data,
            )
        )
        return r.json()

    async def _paginate[T: WeblateComponentSchema | WeblateUnitSchema](
        self, request: WeblateRequestSchema, kind: type[T]
    ) -> list[T]:
        base_params = request.params or WeblateRequestParamsSchema()
        page_model: type[WeblatePageSchema[T]] = cast(Any, WeblatePageSchema)[kind]

        async def fetch_page(page: int) -> WeblatePageSchema[T]:
            async with self._paginate_sem:
                paged = request.model_copy(
                    update={"params": base_params.model_copy(update={"page": page})}
                )
                r = await self._request(paged)
                return page_model.model_validate(r.json())

        page1 = await fetch_page(1)
        actual_size = len(page1.results)
        total_page = math.ceil(page1.count / actual_size) if page1.count else 0

        if total_page <= 1:
            return page1.results

        fetched = await asyncio.gather(
            *[fetch_page(page) for page in range(2, total_page + 1)],
        )
        return [
            *page1.results,
            *[result for page in fetched for result in page.results],
        ]

    async def _request(
        self, request: WeblateRequestSchema, timeout: float | None = None
    ) -> Response:
        params = (
            request.params.model_dump(exclude_none=True) if request.params else None
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._request_sem:
                    r = await self._client.request(
                        request.method,
                        request.path,
                        params=params,
                        json=request.json_body,
                        data=request.data,
                        files=request.files,
                        timeout=timeout or HTTP_TIMEOUT,
                    )
                    if r.status_code == 429 and attempt < RETRY_MAX_ATTEMPTS:
                        retry_after = float(r.headers.get("Retry-After", "1"))
                        logger.warning(
                            f"Weblate 429 on {request.method} {request.path}; "
                            f"sleeping {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    if 500 <= r.status_code < 600 and attempt < RETRY_MAX_ATTEMPTS:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.warning(
                            f"Weblate {r.status_code} on {request.method} {request.path}; "
                            f"retry {attempt}/{RETRY_MAX_ATTEMPTS} after {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    return self._response_or_raise(r)
            except TransportError as exc:
                if attempt >= RETRY_MAX_ATTEMPTS:
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Weblate network error on {request.method} {request.path}: {exc!r}; "
                    f"retry {attempt}/{RETRY_MAX_ATTEMPTS} after {delay}s"
                )
                await asyncio.sleep(delay)
                continue

    def _response_or_raise[T: Response](self, response: T) -> T:
        if response.is_success:
            return response
        raise WeblateAPIError(response.status_code, response.text[:500])
