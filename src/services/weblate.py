import time
from collections.abc import Iterator
from typing import Any, Final

import httpx
from loguru import logger

from src.models.weblate import WeblateConfigSchema

RATE_LIMIT_FLOOR: Final[int] = 100
RETRY_MAX_ATTEMPTS: Final[int] = 3
RETRY_BASE_DELAY: Final[float] = 1.0
HTTP_TIMEOUT: Final[float] = 30.0
HTTP_UPLOAD_TIMEOUT: Final[float] = 300.0
LOCK_BUSY_BASE_DELAY: Final[float] = 60.0
LOCK_BUSY_ERROR_SUBSTRING: Final[str] = "could not be acquired"
TASK_POLL_INTERVAL: Final[float] = 2.0
TASK_POLL_TIMEOUT: Final[float] = 300.0


class WeblateAPIError(Exception):
    """Raised when the Weblate API returns an unrecoverable error."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class WeblateClient:
    """Thin Weblate REST API client.

    Responsibilities: authentication, pagination, rate-limit backoff,
    retry on 5xx / 429, background task polling.

    Non-responsibilities: CSV generation, unit mapping, business rules
    (those live in CorpusConverter / cli/app.py).
    """

    def __init__(self, config: WeblateConfigSchema) -> None:
        self.config = config
        self.base_url = config.url.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Token {config.token}",
                "Accept": "application/json",
            },
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WeblateClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get_project(self) -> dict[str, Any] | None:
        r = self._request("GET", f"projects/{self.config.project_slug}/")
        if r.status_code == 404:
            return None
        self._raise_for_status(r)
        return r.json()

    def create_project(self, name: str, slug: str) -> dict[str, Any]:
        r = self._request(
            "POST",
            "projects/",
            json={"name": name, "slug": slug, "web": "https://example.com/"},
        )
        self._raise_for_status(r)
        return r.json()

    def list_components(self) -> list[dict[str, Any]]:
        return list(self._paginate(f"projects/{self.config.project_slug}/components/"))

    def get_component(self, slug: str) -> dict[str, Any] | None:
        r = self._request("GET", f"components/{self.config.project_slug}/{slug}/")
        if r.status_code == 404:
            return None
        self._raise_for_status(r)
        return r.json()

    def create_component(
        self,
        name: str,
        slug: str,
        csv_bytes: bytes,
        source_language: str = "en",
        is_glossary: bool = False,
        license: str = "",
        license_url: str = "",
        manage_units: bool = False,
        edit_template: bool = False,
    ) -> dict[str, Any]:
        """Create a component by uploading a CSV docfile.

        Internally polls the resulting task URL until the component is ready
        or TASK_POLL_TIMEOUT elapses.

        Post-create PATCHes (Weblate drops these fields silently on POST):
            - `license` / `license_url` — SPDX identifier + canonical URL
            - `manage_units` — required for `create_unit` on non-glossary
              bilingual components; glossary components default to True
              automatically
            - `edit_template` — in addition to manage_units, bilingual CSV
              components need this to accept `create_unit` calls that add
              new source strings to the en.csv template

        The pair (manage_units=True, edit_template=True) was identified
        empirically on hosted.weblate.org 2026-04: without edit_template,
        create_unit returns HTTP 403 "Adding strings is disabled in the
        component configuration" even when manage_units is True. Glossary
        components do not need these PATCHes (Weblate sets them implicitly).
        """
        files = {"docfile": (f"{slug}.csv", csv_bytes, "text/csv")}
        data = {
            "name": name,
            "slug": slug,
            "file_format": "csv",
            "source_language": source_language,
            "new_lang": "add",
            "is_glossary": "true" if is_glossary else "false",
        }
        r = self._request(
            "POST",
            f"projects/{self.config.project_slug}/components/",
            data=data,
            files=files,
            timeout=HTTP_UPLOAD_TIMEOUT,
        )
        self._raise_for_status(r)
        body = r.json()

        task_url = body.get("task_url")
        if task_url:
            self._wait_for_task(task_url)

        # Fields POST silently drops — patch them back on in one go.
        patch_data: dict[str, Any] = {}
        if license:
            patch_data["license"] = license
            if license_url:
                patch_data["license_url"] = license_url
        if manage_units:
            patch_data["manage_units"] = True
        if edit_template:
            patch_data["edit_template"] = True
        if patch_data:
            try:
                body = self.patch_component(slug, patch_data)
            except WeblateAPIError as e:
                logger.warning(
                    f"Failed post-create PATCH on {slug}: {e}; "
                    f"fields attempted: {sorted(patch_data)}"
                )
        return body

    def delete_component(self, slug: str, wait: bool = True) -> None:
        """Delete a component from the project.

        Weblate's DELETE is processed asynchronously; by default we poll
        until a subsequent GET on the same slug returns 404 (or the slug
        endpoint disappears from the project's component list). Pass
        `wait=False` to skip polling.
        """
        r = self._request("DELETE", f"components/{self.config.project_slug}/{slug}/")
        if r.status_code not in (200, 202, 204, 404):
            self._raise_for_status(r)
        if not wait:
            return

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            check = self._request(
                "GET", f"components/{self.config.project_slug}/{slug}/"
            )
            if check.status_code == 404:
                return
            time.sleep(1.0)
        logger.warning(f"Component {slug} still present after 30s delete wait")

    def patch_component(self, slug: str, data: dict[str, Any]) -> dict[str, Any]:
        """PATCH an existing component. Used for fields Weblate silently
        drops on the create endpoint — most notably `license`, which must be
        set after creation or the project emits a "missing license" warning.
        """
        r = self._request(
            "PATCH",
            f"components/{self.config.project_slug}/{slug}/",
            json=data,
        )
        self._raise_for_status(r)
        return r.json()

    def create_translation(self, component_slug: str, lang: str) -> None:
        """Add a target language to a component. Idempotent on re-run.

        Weblate's "already exists" response on POST varies by version:
          - "Translation already exists."         (older)
          - "Could not add '{lang}'!"              (newer, observed on
            hosted.weblate.org 2026-04)
        Both are treated as success so upload workflows can be re-run safely.

        Timeout note: creating a translation for a large component is
        O(unit_count) on the server — Weblate materializes one empty
        target unit per source unit. For the 26K-unit merged base-game
        component this exceeds the default 30s read timeout on
        self-hosted instances, so we pass the longer upload timeout.
        """
        r = self._request(
            "POST",
            f"components/{self.config.project_slug}/{component_slug}/translations/",
            json={"language_code": lang},
            timeout=HTTP_UPLOAD_TIMEOUT,
        )
        if r.status_code in (200, 201):
            return
        if r.status_code == 400:
            body_lower = r.text.lower()
            if "already exists" in body_lower or "could not add" in body_lower:
                logger.debug(f"Translation {lang} for {component_slug} already exists")
                return
        self._raise_for_status(r)

    def upload_file(
        self,
        component_slug: str,
        lang: str,
        csv_bytes: bytes,
        method: str = "translate",
    ) -> dict[str, Any]:
        files = {"file": (f"{component_slug}.csv", csv_bytes, "text/csv")}
        data = {"method": method}
        r = self._request(
            "POST",
            f"translations/{self.config.project_slug}/{component_slug}/{lang}/file/",
            data=data,
            files=files,
            timeout=HTTP_UPLOAD_TIMEOUT,
        )
        self._raise_for_status(r)
        return r.json()

    def download_file(self, component_slug: str, lang: str) -> bytes:
        r = self._request(
            "GET",
            f"translations/{self.config.project_slug}/{component_slug}/{lang}/file/",
            params={"format": "csv"},
        )
        self._raise_for_status(r)
        return r.content

    def list_units(
        self,
        component_slug: str,
        lang: str,
        q: str | None = None,
        page_size: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """Stream every unit in a (component, lang) pair, optionally filtered by `q`.

        `page_size` defaults to 1000 to minimize round trips on large
        components — Weblate's DRF pagination will silently clamp to the
        instance's `MAX_PAGE_SIZE` (100 on most self-hosted setups, 1000
        on hosted.weblate.org), so requesting more than supported is
        harmless. A 10k-unit glossary drops from ~500 sequential page
        requests (default 20/page) to 10 or fewer.
        """
        path = f"translations/{self.config.project_slug}/{component_slug}/{lang}/units/"
        params: dict[str, Any] = {"page_size": page_size}
        if q:
            params["q"] = q
        yield from self._paginate(path, params=params)

    def patch_unit(self, unit_id: int, data: dict[str, Any]) -> dict[str, Any]:
        r = self._request("PATCH", f"units/{unit_id}/", json=data)
        self._raise_for_status(r)
        return r.json()

    def create_unit(
        self,
        component_slug: str,
        lang: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Add a new translation unit to an existing component.

        This is the only incremental-update path for bilingual CSV
        components — `upload_file` with `method="translate"` silently
        skips units whose source string doesn't already exist, and
        `method="replace"` is destructive against translator edits.

        The request body is passed through verbatim so the caller can
        target either monolingual (`key` + `value`) or bilingual /
        glossary (`source` + `target`) shapes. Weblate glossary
        components accept `source`/`target`; regular bilingual CSV
        components accept the same fields in Weblate 4.5+. If a
        deployment rejects these calls, the caller should fall back to
        download → merge → replace-upload.

        Args:
            component_slug: Target component slug (e.g. `{namespace}-XComGame`
                or `glossary-{namespace}`).
            lang: Translation-language code. The unit is language-scoped
                on the URL but visible across all languages once created.
            body: Raw JSON body for the POST request.

        Returns:
            The newly-created unit payload from Weblate.
        """
        r = self._request(
            "POST",
            f"translations/{self.config.project_slug}/{component_slug}/{lang}/units/",
            json=body,
        )
        self._raise_for_status(r)
        return r.json()

    def list_units_page(
        self,
        component_slug: str,
        lang: str,
        q: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> tuple[int, list[dict[str, Any]]]:
        path = f"translations/{self.config.project_slug}/{component_slug}/{lang}/units/"
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if q:
            params["q"] = q
        r = self._request("GET", path, params=params)
        self._raise_for_status(r)
        body = r.json()
        return body["count"], body["results"]

    def search_units(self, q: str, page_size: int = 50) -> list[dict[str, Any]]:
        """Search units globally across all components."""
        r = self._request("GET", "units/", params={"q": q, "page_size": page_size})
        self._raise_for_status(r)
        return r.json().get("results", [])

    def get_translation(self, component_slug: str, lang: str) -> dict[str, Any]:
        """Get translation stats for a component+language pair."""
        r = self._request(
            "GET",
            f"translations/{self.config.project_slug}/{component_slug}/{lang}/",
        )
        self._raise_for_status(r)
        return r.json()

    def get_task(self, url: str) -> dict[str, Any]:
        r = self._request("GET", url)
        self._raise_for_status(r)
        return r.json()

    def _paginate(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        next_url: str | None = path
        first_params = params
        while next_url:
            r = self._request("GET", next_url, params=first_params)
            first_params = None  # only the first page carries caller params
            self._raise_for_status(r)
            body = r.json()
            yield from body.get("results", [])
            next_url = body.get("next")

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send an HTTP request with rate-limit + 5xx/429 retry.

        `RETRY_MAX_ATTEMPTS` is the total number of HTTP calls on persistent
        failure (1 initial attempt + N-1 retries). The gate uses `<` so that
        attempt N is the last call, matching the constant's name.

        Retry policy:
            - **Network errors** (``httpx.TransportError``): connection
              refused, DNS failure, read/write timeout, pool exhaustion.
              Exponential 1s / 2s / 4s, then re-raise on final attempt.
            - **429**: respect Retry-After header (or 1s default).
            - **500/501/502/503**: exponential 1s / 2s / 4s. The server
              errored out fast and nothing is still running.
            - **504 Gateway Timeout** and **component-update lock-busy
              400**: exponential 60s / 120s. A 504 from the reverse
              proxy doesn't mean the server gave up — it means the
              proxy did, while Weblate's import pipeline is still
              holding the `component-update` lock. A short retry hits
              the same lock and returns 400 "Lock ... could not be
              acquired in 5s". Waiting the full proxy-timeout budget
              lets the in-flight import finish; then
              `upload_file(method='add')` is idempotent so the retry
              either no-ops (prior work landed) or finally succeeds.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                r = self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if attempt < RETRY_MAX_ATTEMPTS:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Weblate network error on {method} {url}: {exc!r}; "
                        f"retry {attempt}/{RETRY_MAX_ATTEMPTS} after {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise
            self._respect_rate_limit(r)

            if r.status_code == 429 and attempt < RETRY_MAX_ATTEMPTS:
                retry_after = float(r.headers.get("Retry-After", "1"))
                logger.warning(
                    f"Weblate 429 on {method} {url}; sleeping {retry_after}s"
                )
                time.sleep(retry_after)
                continue
            if self._is_lock_busy(r) and attempt < RETRY_MAX_ATTEMPTS:
                delay = LOCK_BUSY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Weblate {r.status_code} (lock busy / upstream timeout) "
                    f"on {method} {url}; retry {attempt}/{RETRY_MAX_ATTEMPTS} "
                    f"after {delay:.0f}s (in-flight request may still be "
                    "completing)"
                )
                time.sleep(delay)
                continue
            if 500 <= r.status_code < 600 and attempt < RETRY_MAX_ATTEMPTS:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Weblate {r.status_code} on {method} {url}; "
                    f"retry {attempt}/{RETRY_MAX_ATTEMPTS} after {delay}s"
                )
                time.sleep(delay)
                continue
            return r

    @staticmethod
    def _is_lock_busy(response: httpx.Response) -> bool:
        """True when the response signals `wait for an in-flight task`.

        Four distinct server conditions share the same recovery
        strategy (long wait, then retry):

          - **504 Gateway Timeout** — the reverse proxy gave up on an
            upload while Weblate was still importing.
          - **524 A Timeout Occurred** — Cloudflare's equivalent of 504.
            Fires when origin processing exceeds Cloudflare's fixed
            100s upload window. Same recovery: the origin may still be
            working.
          - **423 Locked** — Weblate's repository lock is held by a
            background task (e.g. unit materialization after
            `create_component`). Observed on hosted.weblate.org when
            `create_translation` fires before the component's initial
            import task has finished.
          - **400 with body containing "could not be acquired"** —
            Weblate's `component-update` lock is held by a still-running
            request; the 5s internal acquisition window expired.
        """
        if response.status_code in (504, 524, 423):
            return True
        return (
            response.status_code == 400 and LOCK_BUSY_ERROR_SUBSTRING in response.text
        )

    def _respect_rate_limit(self, response: httpx.Response) -> None:
        remaining_header = response.headers.get("X-RateLimit-Remaining")
        if remaining_header is None:
            return
        try:
            remaining = int(remaining_header)
        except ValueError:
            return
        if remaining >= RATE_LIMIT_FLOOR:
            return
        reset = response.headers.get("X-RateLimit-Reset")
        if not reset:
            return
        try:
            reset_ts = float(reset)
        except ValueError:
            return
        sleep_for = max(reset_ts - time.time(), 0)
        if sleep_for > 0:
            logger.warning(
                f"Weblate rate limit low ({remaining}); sleeping {sleep_for:.1f}s"
            )
            time.sleep(sleep_for)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        raise WeblateAPIError(response.status_code, response.text[:500])

    def _wait_for_task(self, task_url: str) -> None:
        deadline = time.monotonic() + TASK_POLL_TIMEOUT
        while time.monotonic() < deadline:
            task = self.get_task(task_url)
            if task.get("completed"):
                # Weblate may return {"result": null} for "completed with no
                # payload", so `task.get("result", {})` is not enough — the
                # default only kicks in when the key is absent, not null.
                result = task.get("result") or {}
                if result.get("error"):
                    raise WeblateAPIError(
                        500, f"Component task failed: {result['error']}"
                    )
                return
            time.sleep(TASK_POLL_INTERVAL)
        raise WeblateAPIError(504, f"Component task timeout: {task_url}")
