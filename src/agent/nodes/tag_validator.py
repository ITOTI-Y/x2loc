from __future__ import annotations

import asyncio
from typing import TypedDict

from loguru import logger

from src.agent.config import ConfigSchema
from src.agent.llm import TranslationAgent
from src.agent.nodes._batched import invoke_batched
from src.agent.prompts import format_tag_fix_prompt
from src.core.placeholders import validate_tags
from src.models.agent import (
    NewAgentStateSchema,
    TranslationUnitSchema,
)


class TagValidatorOutputSchema(TypedDict):
    candidates: list[TranslationUnitSchema]


async def tag_validator(
    state: NewAgentStateSchema,
    *,
    agent_config: ConfigSchema,
    llm_slots: asyncio.Semaphore,
    llm: TranslationAgent,
) -> TagValidatorOutputSchema:
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

    fixes = await invoke_batched(
        llm,
        [
            (
                candidate.id,
                format_tag_fix_prompt(
                    source=candidate.source,
                    translation=candidate.translated,
                    missing=missing,
                    extra=extra,
                ),
            )
            for candidate, missing, extra in pending
        ],
        units_per_request=agent_config.units_per_request,
        slots=llm_slots,
        label="Tag fix",
    )
    for candidate, _missing, _extra in pending:
        fix = fixes.get(candidate.id)
        if fix is not None and fix.result:
            passed, _, _ = validate_tags(candidate.source, fix.result)
            results.append(
                candidate.model_copy(
                    update={"translated": fix.result, "tag_valid": passed}
                )
            )
        else:
            results.append(candidate.model_copy(update={"tag_valid": False}))
    logger.success("Validated {} units", len(results))
    return {"candidates": results}
