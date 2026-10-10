"""Weblate client tests.

Every request is served in-process by `httpx2.MockTransport`, so the
suite never reaches a live Weblate instance and never mutates a real
project. `FakeWeblate` routes `(method, path)` to canned responses and
records what the client actually sent, which is what most assertions
here inspect.
"""

import asyncio
import json
import ssl
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest
from httpx2 import ConnectError, MockTransport, Request, Response
from loguru import logger

from src.models.weblate import (
    UNIT_PAGE_SIZE,
    CorpusUnitSchema,
    WeblateComponentDraftSchema,
    WeblateConfigSchema,
    WeblateRequestParamsSchema,
    WeblateUnitPatchSchema,
)
from src.services.weblate import (
    RETRY_MAX_ATTEMPTS,
    AsyncWeblateClient,
    WeblateAPIError,
    units_to_csv,
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
    total = UNIT_PAGE_SIZE + UNIT_PAGE_SIZE // 2
    pages = [
        page_payload([unit_payload(start + i) for i in range(size)], count=total)
        for start, size in (
            (0, UNIT_PAGE_SIZE),
            (UNIT_PAGE_SIZE, total - UNIT_PAGE_SIZE),
        )
    ]
    fake.paginate(UNITS_PATH, pages)

    units = await client.list_units(COMPONENT, LANG)

    assert [u.id for u in units] == list(range(total))
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


async def test_wait_for_translation_units_polls_until_ready(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    path = f"translations/{PROJECT}/{COMPONENT}/{LANG}/"
    fake.route(
        "GET", path, Response(200, json={"total": 0}), Response(200, json={"total": 23})
    )
    await client.wait_for_translation_units(COMPONENT, LANG, expected=23)
    assert len(fake.requests) == 2


async def test_wait_for_translation_units_times_out(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    path = f"translations/{PROJECT}/{COMPONENT}/{LANG}/"
    fake.route("GET", path, Response(200, json={"total": 3}))
    with pytest.raises(WeblateAPIError, match="3/23 units"):
        await client.wait_for_translation_units(
            COMPONENT, LANG, expected=23, timeout=0.1
        )


SOURCE_UNITS_PATH = f"translations/{PROJECT}/{COMPONENT}/en/units/"
SOURCE_FILE_PATH = f"translations/{PROJECT}/{COMPONENT}/en/file/"
TARGET_UNITS_PATH = f"translations/{PROJECT}/{COMPONENT}/{LANG}/units/"
TARGET_FILE_PATH = f"translations/{PROJECT}/{COMPONENT}/{LANG}/file/"


def corpus_units(*contexts: str) -> list[CorpusUnitSchema]:
    return [
        CorpusUnitSchema(context=context, source=f"S {context}", target="", note="")
        for context in contexts
    ]


TRANSLATIONS_PATH = f"components/{PROJECT}/{COMPONENT}/translations/"
TARGET_STATS_PATH = f"translations/{PROJECT}/{COMPONENT}/{LANG}/"


def stats_payload(language: str, total: int, translated: int = 0) -> dict[str, Any]:
    return {"language_code": language, "total": total, "translated": translated}


def route_translations(fake: FakeWeblate, *translations: dict[str, Any]) -> None:
    """Serve the component's translation counts; none at all means 404."""
    if not translations:
        fake.route("GET", TRANSLATIONS_PATH, Response(404))
        return
    fake.route(
        "GET",
        TRANSLATIONS_PATH,
        Response(200, json=page_payload(list(translations), len(translations))),
    )


def csv_response(*contexts: str, target: str = "") -> Response:
    units = [
        CorpusUnitSchema(context=c, source=f"S {c}", target=target, note="")
        for c in contexts
    ]
    # A source file carries the source text in its target column.
    content = "target" if target else "source"
    return Response(200, content=units_to_csv(units, content=content))


def route_translation_ready(fake: FakeWeblate, total: int) -> None:
    fake.route("POST", TRANSLATIONS_PATH, Response(201, json={}))
    fake.route("GET", TARGET_STATS_PATH, Response(200, json={"total": total}))


async def test_ssl_error_is_retried(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    fake.route(
        "GET",
        COMPONENT_PATH,
        ssl.SSLError(1, "bad record mac"),
        Response(200, json=component_payload(COMPONENT)),
    )

    component = await client.get_component(COMPONENT)

    assert component is not None
    assert len(sleeps) == 1


async def test_sync_recreates_component_without_source_units(
    client: AsyncWeblateClient, fake: FakeWeblate, sleeps: list[float]
) -> None:
    route_translations(fake, stats_payload("en", 0), stats_payload(LANG, 0))
    fake.route("DELETE", COMPONENT_PATH, Response(204))
    fake.route("POST", COMPONENTS_PATH, Response(201, json={}))
    fake.route("GET", COMPONENT_PATH, Response(404))
    fake.route("PATCH", COMPONENT_PATH, Response(200, json={}))
    route_translation_ready(fake, total=2)

    await client.sync_corpus(
        COMPONENT,
        name="n",
        language=LANG,
        units=corpus_units("a", "b"),
        has_existing_target=False,
    )

    sent = [(r.method, r.url.path) for r in fake.requests]
    assert ("DELETE", API_PREFIX + COMPONENT_PATH) in sent
    assert ("POST", API_PREFIX + COMPONENTS_PATH) in sent


async def test_sync_adds_only_missing_source_units(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    # The target lags the source, so presence is read from the source file.
    route_translations(fake, stats_payload("en", 1), stats_payload(LANG, 0))
    fake.route("GET", SOURCE_FILE_PATH, csv_response("a"))
    fake.route("POST", SOURCE_FILE_PATH, Response(200, json={"accepted": 1}))
    route_translation_ready(fake, total=2)

    snapshot = await client.sync_corpus(
        COMPONENT,
        name="n",
        language=LANG,
        units=corpus_units("a", "b"),
        has_existing_target=False,
    )

    uploads = [
        r
        for r in fake.requests
        if r.method == "POST" and r.url.path == API_PREFIX + SOURCE_FILE_PATH
    ]
    assert len(uploads) == 1
    body = uploads[0].content.decode()
    assert '"b","S b"' in body
    assert '"a","S a"' not in body
    assert snapshot is None


async def test_rejection_body_is_logged_with_token_masked(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    fake.route("GET", COMPONENT_PATH, Response(400, text=f"bad field; echoed {TOKEN}"))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        with pytest.raises(WeblateAPIError):
            await client.get_component(COMPONENT)
    finally:
        logger.remove(sink)

    logged = "".join(messages)
    assert "bad field" in logged
    assert TOKEN not in logged


async def test_sync_recreates_translation_that_never_got_units(
    client: AsyncWeblateClient, fake: FakeWeblate, monkeypatch: pytest.MonkeyPatch
) -> None:
    route_translations(fake, stats_payload("en", 1))
    fake.route("GET", SOURCE_FILE_PATH, csv_response("a"))
    fake.route("POST", TRANSLATIONS_PATH, Response(201, json={}))
    fake.route("GET", TARGET_STATS_PATH, Response(200, json={"total": 0}))
    fake.route("DELETE", TARGET_STATS_PATH, Response(204))
    waits: list[int] = []

    async def wait_once_then_ready(*_args: object, expected: int) -> None:
        waits.append(expected)
        if len(waits) == 1:
            raise WeblateAPIError(504, "not ready")

    monkeypatch.setattr(client, "wait_for_translation_units", wait_once_then_ready)

    await client.sync_corpus(
        COMPONENT,
        name="n",
        language=LANG,
        units=corpus_units("a"),
        has_existing_target=False,
    )

    sent = [(r.method, r.url.path) for r in fake.requests]
    assert ("DELETE", API_PREFIX + TARGET_STATS_PATH) in sent
    assert waits == [1, 1]


async def test_sync_uploads_only_targets_weblate_lacks(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    route_translations(fake, stats_payload("en", 3, 3), stats_payload(LANG, 3, 1))
    fake.route("GET", TARGET_FILE_PATH, csv_response("a", "b", "c", target="x"))
    fake.route(
        "GET",
        TARGET_UNITS_PATH,
        Response(
            200,
            json=page_payload(
                [
                    unit_payload(2, context="b", state=10),
                    unit_payload(3, context="c", target=[""], state=0),
                ],
                2,
            ),
        ),
    )
    fake.route("POST", TARGET_FILE_PATH, Response(200, json={"accepted": 2}))
    units = [
        CorpusUnitSchema(context=c, source=f"S {c}", target=f"T {c}", note="")
        for c in "abc"
    ]

    snapshot = await client.sync_corpus(
        COMPONENT, name="n", language=LANG, units=units, has_existing_target=True
    )

    listed = [r for r in fake.requests if r.url.path == API_PREFIX + TARGET_UNITS_PATH]
    assert [r.url.params["q"] for r in listed] == ["state:<translated"]
    uploads = [
        r
        for r in fake.requests
        if r.method == "POST" and r.url.path == API_PREFIX + TARGET_FILE_PATH
    ]
    assert len(uploads) == 1
    body = uploads[0].content.decode()
    assert '"b"' in body and '"c"' in body
    assert '"a"' not in body
    assert snapshot is None


async def test_synced_component_costs_two_reads_and_returns_snapshot(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    route_translations(fake, stats_payload("en", 1, 1), stats_payload(LANG, 1, 1))
    fake.route("GET", TARGET_FILE_PATH, csv_response("a", target="T a"))
    units = [CorpusUnitSchema(context="a", source="S a", target="T a", note="")]

    snapshot = await client.sync_corpus(
        COMPONENT, name="n", language=LANG, units=units, has_existing_target=True
    )

    assert [(r.method, r.url.path) for r in fake.requests] == [
        ("GET", API_PREFIX + TRANSLATIONS_PATH),
        ("GET", API_PREFIX + TARGET_FILE_PATH),
    ]
    assert snapshot == [
        CorpusUnitSchema(context="a", source="S a", target="T a", note="")
    ]


async def test_download_units_parses_translation_csv(
    client: AsyncWeblateClient, fake: FakeWeblate
) -> None:
    payload = (
        '"context","source","target","developer_comments"\r\n'
        '"a","S, a","多\n行","n"\r\n'
        '"b","S b","",""\r\n'
    )
    fake.route("GET", TARGET_FILE_PATH, Response(200, content=payload.encode()))

    assert await client.download_units(COMPONENT, LANG) == [
        CorpusUnitSchema(context="a", source="S, a", target="多\n行", note="n"),
        CorpusUnitSchema(context="b", source="S b", target="", note=""),
    ]
