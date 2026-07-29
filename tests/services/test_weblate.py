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
    WeblateComponentDraftSchema,
    WeblateConfigSchema,
    WeblateRequestParamsSchema,
    WeblateUnitPatchSchema,
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

COMPONENT_PATH = f"components/{PROJECT}/{COMPONENT}/"
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

        `list_units` fetches pages 2..N concurrently, so arrival order is
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
    return {"slug": slug, "name": slug, "file_format": "csv", "manage_units": False}


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
def draft() -> WeblateComponentDraftSchema:
    return WeblateComponentDraftSchema(
        name="Test Component",
        slug="test-component",
        source_csv="source,target\nOK,确定\n".encode(),
    )


async def test_get_component_parses_payload(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(200, json=component_payload(COMPONENT)))

    component = await client.get_component(COMPONENT)

    assert component is not None
    assert component.slug == COMPONENT
    assert fake.requests[0].url.path == API_PREFIX + COMPONENT_PATH


async def test_get_component_returns_none_on_missing(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(404, text="Not found"))

    assert await client.get_component(COMPONENT) is None
    assert len(fake.requests) == 1
    assert sleeps == []


async def test_requests_carry_token_auth(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(200, json=component_payload(COMPONENT)))
    await client.get_component(COMPONENT)
    assert fake.requests[0].headers["Authorization"] == f"Token {TOKEN}"


async def test_base_url_tolerates_missing_trailing_slash(fake: FakeWeblate) -> None:
    config = WeblateConfigSchema(
        url="https://weblate.example.com/api", token=TOKEN, project_slug=PROJECT
    )
    fake.route("GET", COMPONENT_PATH, Response(200, json=component_payload(COMPONENT)))
    async with AsyncWeblateClient(config, transport=MockTransport(fake)) as client:
        await client.get_component(COMPONENT)
    assert fake.requests[0].url.path == API_PREFIX + COMPONENT_PATH


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
        COMPONENT,
        LANG,
        WeblateRequestParamsSchema(page=2, page_size=50, q="state:<20"),
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
    # Weblate 5.x answers with a partial unit body; the client must not
    # depend on any of its fields.
    fake.route(
        "PATCH", "units/7/", Response(200, json={"target": ["确定"], "state": 20})
    )

    await client.patch_unit(7, WeblateUnitPatchSchema(target=["确定"], state=20))

    assert json.loads(fake.requests[0].content) == {"target": ["确定"], "state": 20}


async def test_create_component_uploads_csv_as_multipart(
    client: AsyncWeblateClient, fake: FakeWeblate, draft: WeblateComponentDraftSchema
) -> None:
    fake.route("POST", COMPONENTS_PATH, Response(201, json={"task_url": None}))
    fake.route("PATCH", f"components/{PROJECT}/{draft.slug}/", Response(200, json={}))

    await client.create_component(draft)

    request = fake.requests[0]
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    body = request.content.decode()
    assert 'name="slug"' in body
    assert draft.slug in body
    assert 'name="file_format"' in body
    assert 'name="source_language"' in body
    assert 'name="license"' in body
    assert 'name="docfile"' in body
    assert 'filename="test-component.csv"' in body
    assert "OK,确定" in body

    patch_request = fake.requests[1]
    assert json.loads(patch_request.content) == {
        "manage_units": True,
        "edit_template": True,
    }


async def test_create_component_raises_on_other_errors(
    client: AsyncWeblateClient, fake: FakeWeblate, draft: WeblateComponentDraftSchema
) -> None:
    fake.route("POST", COMPONENTS_PATH, Response(403, text="permission denied"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.create_component(draft)

    assert excinfo.value.status_code == 403


async def test_server_error_retry_backs_off_exponentially(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route(
        "GET",
        COMPONENT_PATH,
        Response(500, text="boom"),
        Response(503, text="unavailable"),
        Response(200, json=component_payload(COMPONENT)),
    )

    component = await client.get_component(COMPONENT)

    assert component is not None and component.slug == COMPONENT
    assert sleeps == [2.0, 4.0]
    assert len(fake.requests) == 3


async def test_server_error_raises_after_max_attempts(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(500, text="still broken"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.get_component(COMPONENT)

    assert excinfo.value.status_code == 500
    assert len(fake.requests) == RETRY_MAX_ATTEMPTS
    assert sleeps == [2.0, 4.0, 8.0]


async def test_transport_error_is_retried(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route(
        "GET",
        COMPONENT_PATH,
        ConnectError("connection reset"),
        Response(200, json=component_payload(COMPONENT)),
    )

    component = await client.get_component(COMPONENT)

    assert component is not None and component.slug == COMPONENT
    assert sleeps == [2.0]


async def test_transport_error_propagates_after_max_attempts(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", COMPONENT_PATH, ConnectError("host unreachable"))

    with pytest.raises(ConnectError):
        await client.get_component(COMPONENT)

    assert len(fake.requests) == RETRY_MAX_ATTEMPTS
    assert sleeps == [2.0, 4.0, 8.0]


async def test_client_error_is_not_retried(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(403, text="permission denied"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.get_component(COMPONENT)

    assert excinfo.value.status_code == 403
    assert len(fake.requests) == 1
    assert sleeps == []


async def test_error_message_excludes_response_body(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(400, text="token-echoing body"))

    with pytest.raises(WeblateAPIError) as excinfo:
        await client.get_component(COMPONENT)

    assert excinfo.value.status_code == 400
    assert "token-echoing body" not in str(excinfo.value)


async def test_closed_client_rejects_further_requests(
    weblate_config: WeblateConfigSchema, fake: FakeWeblate
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(200, json=component_payload(COMPONENT)))
    client = AsyncWeblateClient(weblate_config, transport=MockTransport(fake))
    await client.close()

    with pytest.raises(RuntimeError):
        await client.get_component(COMPONENT)
