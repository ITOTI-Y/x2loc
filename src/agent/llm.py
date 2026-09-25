from __future__ import annotations

from typing import Final

import httpx
from httpx import AsyncClient
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIStatusError

from src.agent.config import ConfigSchema
from src.agent.prompts import (
    scoring_system_blocks,
    tag_fix_system_blocks,
    translation_system_blocks,
)
from src.models.agent import (
    AgentInputSchema,
    ScoreResultSchema,
    StructuredAgentResponseSchema,
    TranslationOutputSchema,
)

type TranslationAgent = Runnable[
    AgentInputSchema,
    StructuredAgentResponseSchema[TranslationOutputSchema],
]
type ScoringAgent = Runnable[
    AgentInputSchema,
    StructuredAgentResponseSchema[ScoreResultSchema],
]

FATAL_LLM_STATUS: Final = frozenset({400, 401, 403, 404})
LLM_MAX_RETRIES: Final = 4
LLM_TIMEOUT_SECONDS: Final = 45.0


def raise_if_fatal_llm_error(exc: BaseException) -> None:
    """Re-raise configuration-class LLM failures instead of degrading them.

    Auth, permission, unknown-model and malformed-request errors fail every
    retry identically, so retrying them only burns quality-gate rounds.
    Timeouts, rate limits and 5xx are retried by the SDK first; the ones
    that outlast LLM_MAX_RETRIES become empty results for the quality gate.
    """
    if isinstance(exc, APIStatusError) and exc.status_code in FATAL_LLM_STATUS:
        raise exc


def _chat_model(
    *,
    model: str,
    config: ConfigSchema,
    temperature: float,
    http_async_client: AsyncClient | None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=temperature,
        max_completion_tokens=4096,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        http_async_client=http_async_client,
    )


def build_translator_llm(
    config: ConfigSchema,
    *,
    http_async_client: AsyncClient | None = None,
) -> TranslationAgent:
    llm = _chat_model(
        model=config.translation_model_name,
        config=config,
        temperature=config.translation_temperature,
        http_async_client=http_async_client,
    )
    system_blocks = translation_system_blocks(config.target_lang)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=SystemMessage(content=[dict(system_blocks)]),
        response_format=ToolStrategy(TranslationOutputSchema, handle_errors=False),
    )


def build_tag_validator_llm(
    config: ConfigSchema,
    *,
    http_async_client: AsyncClient | None = None,
) -> TranslationAgent:
    llm = _chat_model(
        model=config.effective_validate_model,
        config=config,
        temperature=config.validate_temperature,
        http_async_client=http_async_client,
    )
    system_blocks = tag_fix_system_blocks(config.target_lang)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=SystemMessage(content=[dict(system_blocks)]),
        response_format=ToolStrategy(TranslationOutputSchema, handle_errors=False),
    )


def build_scorer_llm(
    config: ConfigSchema,
    *,
    http_async_client: AsyncClient | None = None,
) -> ScoringAgent:
    llm = _chat_model(
        model=config.effective_scoring_model,
        config=config,
        temperature=config.scoring_temperature,
        http_async_client=http_async_client,
    )
    system_blocks = scoring_system_blocks(config.target_lang)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=SystemMessage(content=[dict(system_blocks)]),
        response_format=ToolStrategy(ScoreResultSchema, handle_errors=False),
    )


def build_llm_http_client() -> httpx.AsyncClient:
    """Shared LLM transport for every job's ChatOpenAI instances.

    Every LLM call goes through `abatch`, so only the async transport is
    wired; the SDK's implicit sync client is never used.

    `trust_env=False` ignores proxy environment variables and
    `follow_redirects=False` refuses 30x, so neither can steer an outbound
    call away from the caller-supplied LLM endpoint.
    """
    timeout = httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=10.0)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)
