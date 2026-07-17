from __future__ import annotations

from typing import TypedDict

from langchain_core.runnables import Runnable
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.prompts import TAG_FIX_TEMPLATE
from src.agent.tools import validate_tags
from src.models.agent import NewAgentStateSchema, TranslationUnitSchema


class TagValidatorOutputSchema(TypedDict):
    candidates: list[TranslationUnitSchema]


async def tag_validator(
    state: NewAgentStateSchema, *, agent_config: AgentConfigSchema, llm: Runnable
) -> TagValidatorOutputSchema:
    def _build_validate_input(
        c: TranslationUnitSchema, missing: dict[str, int], extra: dict[str, int]
    ) -> dict:
        fix_prompt = TAG_FIX_TEMPLATE.format(
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
        config={"max_concurrency": 10},
    )
    for response in responses:
        if response is not None:
            structured = response.get("structured_response")
            if isinstance(structured, TranslationUnitSchema):
                passed, _, _ = validate_tags(structured.source, structured.translated)
                if passed:
                    results.append(structured.model_copy(update={"tag_valid": True}))
                else:
                    results.append(structured.model_copy(update={"tag_valid": False}))
    if len(results) != len(state.candidates):
        logger.warning(
            f"Tag validator failed to fix {len(state.candidates) - len(results)} units"
        )
    logger.success(f"Validated {len(results)} units")
    return {"candidates": results}
