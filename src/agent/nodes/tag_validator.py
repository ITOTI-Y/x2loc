from __future__ import annotations

from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.llm import TranslationAgent, raise_if_fatal_llm_error
from src.agent.prompts import format_tag_fix_prompt
from src.agent.tools import validate_tags
from src.models.agent import (
    AgentInputSchema,
    NewAgentStateSchema,
    TranslationOutputSchema,
    TranslationUnitSchema,
)


class TagValidatorOutputSchema(TypedDict):
    candidates: list[TranslationUnitSchema]


async def tag_validator(
    state: NewAgentStateSchema,
    *,
    agent_config: ConfigSchema,
    llm: TranslationAgent,
) -> TagValidatorOutputSchema:
    def build_input(
        candidate: TranslationUnitSchema,
        missing: dict[str, int],
        extra: dict[str, int],
    ) -> AgentInputSchema:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": format_tag_fix_prompt(
                        source=candidate.source,
                        translation=candidate.translated,
                        missing=missing,
                        extra=extra,
                    ),
                }
            ]
        }

    results: list[TranslationUnitSchema] = []
    pending: list[tuple[TranslationUnitSchema, dict[str, int], dict[str, int]]] = []
    for candidate in state.candidates:
        if not candidate.translated.strip():
            results.append(candidate.model_copy(update={"tag_valid": False}))
            continue
        passed, missing, extra = validate_tags(candidate.source, candidate.translated)
        if passed:
            results.append(candidate.model_copy(update={"tag_valid": True}))
        else:
            pending.append((candidate, missing, extra))

    responses = await llm.abatch(
        [
            build_input(candidate, missing, extra)
            for candidate, missing, extra in pending
        ],
        config={"max_concurrency": agent_config.max_concurrency},
        return_exceptions=True,
    )
    for response, (candidate, _missing, _extra) in zip(responses, pending, strict=True):
        if isinstance(response, BaseException):
            raise_if_fatal_llm_error(response)
            logger.warning(
                f"Tag fix request failed for unit {candidate.id}: {response!r}"
            )
            results.append(candidate.model_copy(update={"tag_valid": False}))
            continue
        structured = response.get("structured_response") if response else None
        if isinstance(structured, TranslationOutputSchema) and structured.result:
            passed, _, _ = validate_tags(candidate.source, structured.result)
            results.append(
                candidate.model_copy(
                    update={"translated": structured.result, "tag_valid": passed}
                )
            )
        else:
            results.append(candidate.model_copy(update={"tag_valid": False}))
    logger.success("Validated {} units", len(results))
    return {"candidates": results}
