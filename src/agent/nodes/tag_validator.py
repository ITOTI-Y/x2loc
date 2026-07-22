from __future__ import annotations

from typing import TypedDict

from langchain_core.runnables import Runnable
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.prompts import format_tag_fix_prompt
from src.agent.tools import validate_tags
from src.models.agent import (
    NewAgentStateSchema,
    TranslationOutputSchema,
    TranslationUnitSchema,
)


class TagValidatorOutputSchema(TypedDict):
    candidates: list[TranslationUnitSchema]


async def tag_validator(
    state: NewAgentStateSchema, *, agent_config: AgentConfigSchema, llm: Runnable
) -> TagValidatorOutputSchema:
    def _build_validate_input(
        c: TranslationUnitSchema, missing: dict[str, int], extra: dict[str, int]
    ) -> dict:
        fix_prompt = format_tag_fix_prompt(
            source=c.source,
            translation=c.translated,
            missing=missing,
            extra=extra,
        )
        return {
            "messages": [
                {
                    "role": "user",
                    "content": fix_prompt,
                }
            ]
        }

    results: list[TranslationUnitSchema] = []
    to_validate: list[tuple[TranslationUnitSchema, dict, dict]] = []
    for c in state.candidates:
        if not c.translated.strip():
            results.append(c.model_copy(update={"tag_valid": False}))
            continue
        passed, missing, extra = validate_tags(c.source, c.translated)
        if passed:
            results.append(c.model_copy(update={"tag_valid": True}))
            continue
        to_validate.append((c, missing, extra))

    inputs = [
        _build_validate_input(c, missing, extra) for c, missing, extra in to_validate
    ]
    responses = await llm.abatch(
        inputs=inputs,
    )
    fixed_count = 0
    for response, (c, _, _) in zip(responses, to_validate, strict=True):
        structured = response.get("structured_response") if response else None
        if isinstance(structured, TranslationOutputSchema) and structured.result:
            passed, _, _ = validate_tags(c.source, structured.result)
            fixed_count += passed
            results.append(
                c.model_copy(
                    update={"translated": structured.result, "tag_valid": passed}
                )
            )
        else:
            results.append(c.model_copy(update={"tag_valid": False}))
    if fixed_count < len(to_validate):
        logger.warning(
            f"Tag validator failed to fix {len(to_validate) - fixed_count} units"
        )
    logger.success(f"Validated {len(results)} units")
    return {"candidates": results}
