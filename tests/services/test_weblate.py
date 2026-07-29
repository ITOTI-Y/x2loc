"""Weblate client tests.

Every request is served in-process by `httpx2.MockTransport`, so the
suite never reaches a live Weblate instance and never mutates a real
project. `FakeWeblate` routes `(method, path)` to canned responses and
records what the client actually sent, which is what most assertions
here inspect.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest
from httpx2 import ConnectError, MockTransport, Request, Response

from src.models.weblate import (
    WeblateComponentSchema,
    WeblateConfigSchema,
    WeblateLanguageSchema,
    WeblateRequestParamsSchema,
)
from src.services.weblate import (
    RETRY_MAX_ATTEMPTS,
    AsyncWeblateClient,
    WeblateAPIError,
)

BASE_URL = "https://weblate.example.com/api/"
API_PREFIX = "/api/"
PROJECT = "xcom2-test"
TOKEN = "wlp_test_token"
COMPONENT = "mod-42-abc"
LANG = "zh_Hans"

PROJECT_PATH = f"projects/{PROJECT}/"
COMPONENTS_PATH = f"projects/{PROJECT}/components/"
UNITS_PATH = f"translations/{PROJECT}/{COMPONENT}/{LANG}/units/"


class FakeWeblate:
    """In-process Weblate stand-in driving `httpx2.MockTransport`."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], Callable[[Request], Response]] = {}
        self.requests: list[Request] = []

    def route(self, method: str, path: str, *responses: Response | Exception) -> None:
        """Serve `responses` in order; the last one sticks for later calls."""
        queue = list(responses)

        def serve(_request: Request) -> Response:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item

        self._routes[(method, API_PREFIX + path)] = serve

    def paginate(self, path: str, pages: list[dict[str, Any]]) -> None:
        """Serve page N by the request's `page` param, not by call order.

        `_paginate` fetches pages 2..N concurrently, so arrival order is
        undefined and a plain response queue would hand back the wrong page.
        """

        def serve(request: Request) -> Response:
            page = int(request.url.params.get("page", 1))
            return Response(200, json=pages[page - 1])

        self._routes[("GET", API_PREFIX + path)] = serve

    def __call__(self, request: Request) -> Response:
        self.requests.append(request)
        serve = self._routes.get((request.method, request.url.path))
        if serve is None:
            raise AssertionError(
                f"unrouted request: {request.method} {request.url.path}"
            )
        return serve(request)


def page_payload(
    results: list[dict[str, Any]], count: int, next_url: str | None = None
) -> dict[str, Any]:
    return {"results": results, "count": count, "next": next_url}


def unit_payload(unit_id: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": unit_id,
        "translation": f"{BASE_URL}translations/{PROJECT}/{COMPONENT}/{LANG}/",
        "language_code": LANG,
        "source": ["OK"],
        "target": ["确定"],
        "context": f"key-{unit_id}",
        "source_unit": f"{BASE_URL}units/{unit_id}/",
        "web_url": f"https://weblate.example.com/translate/{PROJECT}/",
        "url": f"{BASE_URL}units/{unit_id}/",
        "position": unit_id,
    }
    payload.update(overrides)
    return payload


def component_payload(slug: str) -> dict[str, Any]:
    return {
        "name": slug,
        "slug": slug,
        "source_language": {"id": 1, "code": "en", "name": "English"},
    }


@pytest.fixture
def weblate_config() -> WeblateConfigSchema:
    return WeblateConfigSchema(url=BASE_URL, token=TOKEN, project_slug=PROJECT)


@pytest.fixture
def fake() -> FakeWeblate:
    return FakeWeblate()


@pytest.fixture
async def client(
    weblate_config: WeblateConfigSchema, fake: FakeWeblate
) -> AsyncGenerator[AsyncWeblateClient]:
    async with AsyncWeblateClient(
        weblate_config, transport=MockTransport(fake)
    ) as client:
        yield client


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry backoff delays instead of actually waiting them out."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


@pytest.fixture
def component() -> WeblateComponentSchema:
    return WeblateComponentSchema(
        name="Test Component",
        slug="test-component",
        source_csv="source,target\nOK,确定\n".encode(),
        source_language=WeblateLanguageSchema(id=1, code="en", name="English"),
    )


async def test_get_project_returns_payload(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", PROJECT_PATH, Response(200, json={"slug": PROJECT}))
    assert await client.get_project() == {"slug": PROJECT}
    assert fake.requests[0].url.path == API_PREFIX + PROJECT_PATH


async def test_requests_carry_token_auth(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", PROJECT_PATH, Response(200, json={}))
    await client.get_project()
    assert fake.requests[0].headers["Authorization"] == f"Token {TOKEN}"


async def test_base_url_tolerates_missing_trailing_slash(fake: FakeWeblate) -> None:
    config = WeblateConfigSchema(
        url="https://weblate.example.com/api", token=TOKEN, project_slug=PROJECT
    )
    fake.route("GET", PROJECT_PATH, Response(200, json={}))
    async with AsyncWeblateClient(config, transport=MockTransport(fake)) as client:
        await client.get_project()
    assert fake.requests[0].url.path == API_PREFIX + PROJECT_PATH


async def test_list_components_returns_single_page(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.paginate(
        COMPONENTS_PATH,
        [page_payload([component_payload("a"), component_payload("b")], count=2)],
    )
    components = await client.list_components()
    assert [c.slug for c in components] == ["a", "b"]
    assert len(fake.requests) == 1


async def test_list_components_aggregates_all_pages(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    pages = [
        page_payload(
            [component_payload(f"c-{start + i}") for i in range(size)], count=250
        )
        for start, size in ((0, 100), (100, 100), (200, 50))
    ]
    fake.paginate(COMPONENTS_PATH, pages)

    components = await client.list_components()

    assert [c.slug for c in components] == [f"c-{i}" for i in range(250)]
    assert sorted(int(r.url.params["page"]) for r in fake.requests) == [1, 2, 3]


async def test_list_units_aggregates_all_pages(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    pages = [
        page_payload([unit_payload(start + i) for i in range(size)], count=150)
        for start, size in ((0, 100), (100, 50))
    ]
    fake.paginate(UNITS_PATH, pages)

    units = await client.list_units(COMPONENT, LANG)

    assert [u.id for u in units] == list(range(150))
    assert sorted(int(r.url.params["page"]) for r in fake.requests) == [1, 2]


async def test_list_units_omits_empty_query(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.paginate(UNITS_PATH, [page_payload([unit_payload(1)], count=1)])
    await client.list_units(COMPONENT, LANG)
    assert "q" not in fake.requests[0].url.params


async def test_list_units_page_forwards_params(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", UNITS_PATH, Response(200, json=page_payload([], count=0)))

    page = await client.list_units_page(
        COMPONENT, LANG, page=2, page_size=50, q="state:<20"
    )

    assert page.count == 0
    params = fake.requests[0].url.params
    assert params["page"] == "2"
    assert params["page_size"] == "50"
    assert params["q"] == "state:<20"


async def test_search_units_parses_and_joins_list_fields(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route(
        "GET",
        "units/",
        Response(200, json={"results": [unit_payload(7, source=["A", "B"])]}),
    )

    units = await client.search_units(WeblateRequestParamsSchema(q="foo"))

    assert [u.id for u in units] == [7]
    assert units[0].source == "AB"
    assert units[0].target == "确定"


async def test_search_units_handles_missing_results_key(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", "units/", Response(200, json={}))
    assert await client.search_units(WeblateRequestParamsSchema()) == []


async def test_patch_unit_sends_json_body(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("PATCH", "units/7/", Response(200, json={"id": 7}))

    result = await client.patch_unit(7, {"target": ["确定"], "state": 20})

    assert result == {"id": 7}
    assert json.loads(fake.requests[0].content) == {"target": ["确定"], "state": 20}


async def test_create_component_uploads_csv_as_multipart(
    client: AsyncWeblateClient, fake: FakeWeblate, component: WeblateComponentSchema
) -> None:
    fake.route("POST", COMPONENTS_PATH, Response(201, json={"slug": component.slug}))

    await client.create_component(component=component)

    request = fake.requests[0]
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    body = request.content.decode()
    assert 'name="slug"' in body
    assert component.slug in body
    assert 'name="file_format"' in body
    assert 'filename="test-component.csv"' in body
    assert "OK,确定" in body


async def test_create_component_swallows_duplicate_slug(
    client: AsyncWeblateClient, fake: FakeWeblate, component: WeblateComponentSchema
) -> None:
    fake.route(
        "POST",
        COMPONENTS_PATH,
        Response(400, json={"slug": ["component with this slug already exists"]}),
    )
    await client.create_component(component=component)


async def test_create_component_raises_on_other_errors(
    client: AsyncWeblateClient, fake: FakeWeblate, component: WeblateComponentSchema
) -> None:
    fake.route("POST", COMPONENTS_PATH, Response(403, text="permission denied"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.create_component(component=component)

    assert excinfo.value.status == 403


async def test_delete_component_accepts_no_content(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    path = f"components/{PROJECT}/{COMPONENT}/"
    fake.route("DELETE", path, Response(204))
    await client.delete_component(component_slug=COMPONENT)
    assert fake.requests[0].url.path == API_PREFIX + path


async def test_delete_component_treats_missing_as_deleted(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route(
        "DELETE", f"components/{PROJECT}/{COMPONENT}/", Response(404, text="Not found")
    )
    await client.delete_component(component_slug=COMPONENT)


async def test_delete_component_raises_on_forbidden(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route(
        "DELETE", f"components/{PROJECT}/{COMPONENT}/", Response(403, text="denied")
    )

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.delete_component(component_slug=COMPONENT)

    assert excinfo.value.status == 403


async def test_rate_limit_retry_honours_retry_after(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route(
        "GET",
        PROJECT_PATH,
        Response(429, headers={"Retry-After": "7"}, text="slow down"),
        Response(200, json={"slug": PROJECT}),
    )

    assert await client.get_project() == {"slug": PROJECT}
    assert sleeps == [7.0]
    assert len(fake.requests) == 2


async def test_server_error_retry_backs_off_exponentially(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route(
        "GET",
        PROJECT_PATH,
        Response(500, text="boom"),
        Response(503, text="unavailable"),
        Response(200, json={"slug": PROJECT}),
    )

    assert await client.get_project() == {"slug": PROJECT}
    assert sleeps == [1.0, 2.0]
    assert len(fake.requests) == 3


async def test_server_error_raises_after_max_attempts(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", PROJECT_PATH, Response(500, text="still broken"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.get_project()

    assert excinfo.value.status == 500
    assert "still broken" in excinfo.value.message
    assert len(fake.requests) == RETRY_MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0]


async def test_transport_error_is_retried(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route(
        "GET",
        PROJECT_PATH,
        ConnectError("connection reset"),
        Response(200, json={"slug": PROJECT}),
    )

    assert await client.get_project() == {"slug": PROJECT}
    assert sleeps == [1.0]


async def test_transport_error_propagates_after_max_attempts(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", PROJECT_PATH, ConnectError("host unreachable"))

    with pytest.raises(ConnectError):
        await client.get_project()

    assert len(fake.requests) == RETRY_MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0]


async def test_client_error_is_not_retried(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", PROJECT_PATH, Response(404, text="No Project matches"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.get_project()

    assert excinfo.value.status == 404
    assert "No Project matches" in excinfo.value.message
    assert len(fake.requests) == 1
    assert sleeps == []


async def test_error_message_is_truncated(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", PROJECT_PATH, Response(400, text="x" * 900))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.get_project()

    assert len(excinfo.value.message) == 500


async def test_closed_client_rejects_further_requests(
    weblate_config: WeblateConfigSchema, fake: FakeWeblate
) -> None:
    fake.route("GET", PROJECT_PATH, Response(200, json={}))
    client = AsyncWeblateClient(weblate_config, transport=MockTransport(fake))
    await client.close()

    with pytest.raises(RuntimeError):
        await client.get_project()


@pytest.mark.xfail(
    raises=ZeroDivisionError,
    strict=True,
    reason=(
        "_paginate derives the page count from len(page1.results); a non-zero "
        "count with an empty first page divides by zero"
    ),
)
async def test_paginate_empty_first_page_with_nonzero_count(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.paginate(COMPONENTS_PATH, [page_payload([], count=5)])
    await client.list_components()
