"""LLM HTTP transport, kept free of the LangChain stack so that a run
which never calls an LLM never imports it."""

import httpx


def build_llm_http_client(timeout_seconds: float) -> httpx.AsyncClient:
    """Shared LLM transport for every job's ChatOpenAI instances.

    Every LLM call goes through `abatch`, so only the async transport is
    wired; the SDK's implicit sync client is never used.

    `trust_env=False` ignores proxy environment variables and
    `follow_redirects=False` refuses 30x, so neither can steer an outbound
    call away from the caller-supplied LLM endpoint.
    """
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)
