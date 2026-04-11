"""Unit tests for WeblateClient with respx-mocked HTTP transport."""

import time as _real_time
from typing import Any

import httpx
import pytest
import respx

from src.services.weblate import (
    RETRY_MAX_ATTEMPTS,
    WeblateAPIError,
    WeblateClient,
)

BASE = "https://weblate.example.com/api/"


class FakeTime:
    """Drop-in replacement for the `time` module reference inside weblate.py.

    We inject this via `monkeypatch.setattr("src.services.weblate.time", ...)`
    so only the weblate module's `time` name is rebound — stdlib `time.sleep`
    / `time.time` stay untouched, avoiding global side effects that would
    otherwise break httpx/respx/loguru internals.
    """

    def __init__(self, now: float = 1000.0) -> None:
        self.sleeps: list[float] = []
        self._now = now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def time(self) -> float:
        return self._now

    def monotonic(self) -> float:
        # Real monotonic keeps the `_wait_for_task` deadline realistic.
        return _real_time.monotonic()


@pytest.fixture
def fake_time(monkeypatch: pytest.MonkeyPatch) -> FakeTime:
    ft = FakeTime()
    monkeypatch.setattr("src.services.weblate.time", ft)
    return ft


class TestAuthAndBasicRequests:
    @respx.mock
    def test_auth_header_is_set(self, weblate_client: WeblateClient) -> None:
        route = respx.get(f"{BASE}projects/xcom2-test/").mock(
            return_value=httpx.Response(200, json={"slug": "xcom2-test"})
        )
        weblate_client.get_project()
        assert route.called
        sent = route.calls.last.request
        assert sent.headers["authorization"] == "Token wlp_test_token"

    @respx.mock
    def test_get_project_404_returns_none(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}projects/xcom2-test/").mock(return_value=httpx.Response(404))
        assert weblate_client.get_project() is None

    @respx.mock
    def test_get_project_success(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}projects/xcom2-test/").mock(
            return_value=httpx.Response(
                200, json={"slug": "xcom2-test", "name": "XCOM 2"}
            )
        )
        result = weblate_client.get_project()
        assert result is not None
        assert result["slug"] == "xcom2-test"

    @respx.mock
    def test_unauthorized_raises(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}projects/xcom2-test/").mock(
            return_value=httpx.Response(401, text="bad token")
        )
        with pytest.raises(WeblateAPIError) as exc_info:
            weblate_client.get_project()
        assert exc_info.value.status == 401

    @respx.mock
    def test_create_project(self, weblate_client: WeblateClient) -> None:
        respx.post(f"{BASE}projects/").mock(
            return_value=httpx.Response(
                201, json={"slug": "new-proj", "name": "new-proj"}
            )
        )
        result = weblate_client.create_project("new-proj", "new-proj")
        assert result["slug"] == "new-proj"


class TestComponents:
    @respx.mock
    def test_get_component_404(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}components/xcom2-test/XComGame/").mock(
            return_value=httpx.Response(404)
        )
        assert weblate_client.get_component("XComGame") is None

    @respx.mock
    def test_get_component_success(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}components/xcom2-test/XComGame/").mock(
            return_value=httpx.Response(
                200, json={"slug": "XComGame", "stats": {"total": 3012}}
            )
        )
        comp = weblate_client.get_component("XComGame")
        assert comp is not None
        assert comp["stats"]["total"] == 3012

    @respx.mock
    def test_create_component_with_task_polling(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        del fake_time  # only used to disable real sleep
        task_url = f"{BASE}tasks/deadbeef/"
        respx.post(f"{BASE}projects/xcom2-test/components/").mock(
            return_value=httpx.Response(
                201,
                json={"slug": "XComGame", "task_url": task_url},
            )
        )
        respx.get(task_url).mock(
            side_effect=[
                httpx.Response(200, json={"completed": False}),
                httpx.Response(200, json={"completed": True, "result": {}}),
            ]
        )
        result = weblate_client.create_component(
            name="XComGame",
            slug="XComGame",
            csv_bytes=b"context,source,target,developer_comments\n",
        )
        assert result["slug"] == "XComGame"

    @respx.mock
    def test_create_component_task_result_null(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        """Regression: Weblate may return {"completed": true, "result": null}.

        The happy path must not crash with AttributeError — `task.get(result, {})`
        would return None (not the `{}` default) when the key is explicitly null.
        """
        del fake_time
        task_url = f"{BASE}tasks/null/"
        respx.post(f"{BASE}projects/xcom2-test/components/").mock(
            return_value=httpx.Response(201, json={"slug": "X", "task_url": task_url})
        )
        respx.get(task_url).mock(
            return_value=httpx.Response(200, json={"completed": True, "result": None})
        )
        result = weblate_client.create_component(name="X", slug="X", csv_bytes=b"")
        assert result["slug"] == "X"

    @respx.mock
    def test_create_component_task_failure(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        del fake_time  # only used to disable real sleep
        task_url = f"{BASE}tasks/bad/"
        respx.post(f"{BASE}projects/xcom2-test/components/").mock(
            return_value=httpx.Response(
                201,
                json={"slug": "X", "task_url": task_url},
            )
        )
        respx.get(task_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "completed": True,
                    "result": {"error": "csv parse failure"},
                },
            )
        )
        with pytest.raises(WeblateAPIError):
            weblate_client.create_component(name="X", slug="X", csv_bytes=b"")


class TestTranslationsAndFiles:
    @respx.mock
    def test_create_translation_success(self, weblate_client: WeblateClient) -> None:
        route = respx.post(f"{BASE}components/xcom2-test/XComGame/translations/").mock(
            return_value=httpx.Response(201, json={"language_code": "zh_Hans"})
        )
        weblate_client.create_translation("XComGame", "zh_Hans")
        assert route.called

    @respx.mock
    def test_create_translation_idempotent_on_duplicate_older_phrasing(
        self, weblate_client: WeblateClient
    ) -> None:
        """HTTP 400 with 'already exists' body is swallowed."""
        respx.post(f"{BASE}components/xcom2-test/XComGame/translations/").mock(
            return_value=httpx.Response(400, text="Translation already exists.")
        )
        weblate_client.create_translation("XComGame", "zh_Hans")

    @respx.mock
    def test_create_translation_idempotent_on_could_not_add(
        self, weblate_client: WeblateClient
    ) -> None:
        """HTTP 400 with 'Could not add' body is also treated as idempotent.

        Regression: hosted.weblate.org (~2026-04) returns this phrasing on
        re-add attempts, not the older 'already exists'.
        """
        respx.post(f"{BASE}components/xcom2-test/XComGame/translations/").mock(
            return_value=httpx.Response(
                400,
                json={
                    "type": "validation_error",
                    "errors": [
                        {
                            "code": "invalid",
                            "detail": "Could not add 'zh_Hans'!",
                            "attr": "language_code",
                        }
                    ],
                },
            )
        )
        weblate_client.create_translation("XComGame", "zh_Hans")

    @respx.mock
    def test_upload_file(self, weblate_client: WeblateClient) -> None:
        route = respx.post(
            f"{BASE}translations/xcom2-test/XComGame/zh_Hans/file/"
        ).mock(return_value=httpx.Response(200, json={"accepted": 10}))
        result = weblate_client.upload_file(
            "XComGame", "zh_Hans", b"csv payload", method="translate"
        )
        assert route.called
        assert result == {"accepted": 10}

    @respx.mock
    def test_download_file_returns_bytes(self, weblate_client: WeblateClient) -> None:
        payload = b"context,source,target\nSection::Key,OK,\xe7\xa1\xae\xe5\xae\x9a\n"
        respx.get(f"{BASE}translations/xcom2-test/XComGame/zh_Hans/file/").mock(
            return_value=httpx.Response(
                200, content=payload, headers={"content-type": "text/csv"}
            )
        )
        data = weblate_client.download_file("XComGame", "zh_Hans")
        assert data == payload


class TestPagination:
    @respx.mock
    def test_list_components_follows_next(self, weblate_client: WeblateClient) -> None:
        """Verify `_paginate` follows the `next` link until None.

        We register a single route and use `side_effect` to deliver two
        successive responses. Page 1's `next` points back at the same URL
        (testing the follow-next loop, not URL routing); the second call
        consumes the second response and the loop terminates.
        """
        page1: dict[str, Any] = {
            "results": [{"slug": "a"}, {"slug": "b"}],
            "next": f"{BASE}projects/xcom2-test/components/",
        }
        page2: dict[str, Any] = {
            "results": [{"slug": "c"}],
            "next": None,
        }
        respx.get(f"{BASE}projects/xcom2-test/components/").mock(
            side_effect=[
                httpx.Response(200, json=page1),
                httpx.Response(200, json=page2),
            ]
        )

        slugs = [c["slug"] for c in weblate_client.list_components()]
        assert slugs == ["a", "b", "c"]

    @respx.mock
    def test_pagination_stops_on_null_next(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}projects/xcom2-test/components/").mock(
            return_value=httpx.Response(
                200,
                json={"results": [{"slug": "only"}], "next": None},
            )
        )
        slugs = [c["slug"] for c in weblate_client.list_components()]
        assert slugs == ["only"]

    @respx.mock
    def test_list_units_is_lazy(self, weblate_client: WeblateClient) -> None:
        respx.get(f"{BASE}translations/xcom2-test/glossary/en/units/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"id": 1, "context": "Foo::term"},
                        {"id": 2, "context": "Bar::term"},
                    ],
                    "next": None,
                },
            )
        )
        units = list(weblate_client.list_units("glossary", "en"))
        assert [u["id"] for u in units] == [1, 2]


class TestRetries:
    @respx.mock
    def test_429_retries_after_header_wait(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        respx.get(f"{BASE}projects/xcom2-test/").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, json={"slug": "ok"}),
            ]
        )
        result = weblate_client.get_project()
        assert result == {"slug": "ok"}
        assert fake_time.sleeps == [2.0]

    @respx.mock
    def test_5xx_exponential_backoff(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        respx.get(f"{BASE}projects/xcom2-test/").mock(
            side_effect=[
                httpx.Response(502),
                httpx.Response(503),
                httpx.Response(200, json={"slug": "ok"}),
            ]
        )
        weblate_client.get_project()
        # First failure → 1s, second → 2s
        assert fake_time.sleeps == [1.0, 2.0]

    @respx.mock
    def test_5xx_gives_up_after_max_attempts(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        del fake_time  # only used to disable real sleep
        route = respx.get(f"{BASE}projects/xcom2-test/").mock(
            return_value=httpx.Response(500, text="boom")
        )
        with pytest.raises(WeblateAPIError) as exc_info:
            weblate_client.get_project()
        assert exc_info.value.status == 500
        # RETRY_MAX_ATTEMPTS is the total number of HTTP calls on persistent
        # failure (1 initial + N-1 retries). With `attempt < RETRY_MAX_ATTEMPTS`
        # the Nth call is the last and raises.
        assert route.call_count == RETRY_MAX_ATTEMPTS

    @respx.mock
    def test_rate_limit_header_triggers_preemptive_sleep(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        # fake_time.time() returns 1000.0; reset is 1005 → sleep 5s
        respx.get(f"{BASE}projects/xcom2-test/").mock(
            return_value=httpx.Response(
                200,
                json={"slug": "ok"},
                headers={
                    "X-RateLimit-Remaining": "5",
                    "X-RateLimit-Reset": "1005",
                },
            )
        )
        weblate_client.get_project()
        assert fake_time.sleeps == [pytest.approx(5.0)]

    @respx.mock
    def test_rate_limit_header_above_floor_no_sleep(
        self, weblate_client: WeblateClient, fake_time: FakeTime
    ) -> None:
        respx.get(f"{BASE}projects/xcom2-test/").mock(
            return_value=httpx.Response(
                200,
                json={"slug": "ok"},
                headers={
                    "X-RateLimit-Remaining": "4999",
                    "X-RateLimit-Reset": "9999999999",
                },
            )
        )
        weblate_client.get_project()
        assert fake_time.sleeps == []


class TestUnitPatch:
    @respx.mock
    def test_patch_unit(self, weblate_client: WeblateClient) -> None:
        respx.patch(f"{BASE}units/42/").mock(
            return_value=httpx.Response(
                200, json={"id": 42, "extra_flags": "read-only"}
            )
        )
        result = weblate_client.patch_unit(42, {"extra_flags": "read-only"})
        assert result["extra_flags"] == "read-only"


class TestCreateUnit:
    """Incremental-add path for existing components (Mode 2 upload)."""

    @respx.mock
    def test_create_unit_posts_body_to_lang_units_endpoint(
        self, weblate_client: WeblateClient
    ) -> None:
        route = respx.post(
            f"{BASE}translations/xcom2-test/glossary-1122837889-more-traits/en/units/"
        ).mock(
            return_value=httpx.Response(
                201,
                json={
                    "id": 9001,
                    "source": ["Chain Lightning"],
                    "target": [""],
                    "context": "Chain Lightning::ability",
                },
            )
        )
        body = {
            "source": "Chain Lightning",
            "target": "",
            "context": "Chain Lightning::ability",
            "state": 0,
        }

        result = weblate_client.create_unit(
            "glossary-1122837889-more-traits", "en", body
        )

        assert result["id"] == 9001
        assert route.called
        assert route.calls.last.request.method == "POST"
        # Body was passed through verbatim.
        import json

        sent = json.loads(route.calls.last.request.content)
        assert sent == body

    @respx.mock
    def test_create_unit_raises_on_error(self, weblate_client: WeblateClient) -> None:
        respx.post(f"{BASE}translations/xcom2-test/x/en/units/").mock(
            return_value=httpx.Response(400, json={"detail": "bad"})
        )

        with pytest.raises(WeblateAPIError) as exc:
            weblate_client.create_unit("x", "en", {"source": "bad"})
        assert exc.value.status == 400


class TestContextManager:
    def test_enter_exit_closes_client(self, weblate_config: Any) -> None:
        with WeblateClient(weblate_config) as client:
            assert client._client is not None
        # After exit, the httpx.Client is closed — further requests fail
        with pytest.raises(RuntimeError):
            client._client.request("GET", "projects/")
