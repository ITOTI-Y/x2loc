from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from loguru import logger

from src.agent._share import MAX_TAG_RETRIES
from src.agent.config import AgentConfigSchema
from src.agent.llm import build_tag_validator_llm
from src.agent.prompts import TAG_FIX_TEMPLATE
from src.agent.state import AgentState, TranslationCandidate
from src.agent.tools import validate_tags


async def tag_validator(state: AgentState, *, agent_config: AgentConfigSchema) -> dict:
    tag_validator_llm = build_tag_validator_llm(agent_config)

    async def _validate_one(c: TranslationCandidate) -> TranslationCandidate:
        if c["pattern_matched"]:
            return c

        passed, missing, extra = validate_tags(c["source"], c["translation"])
        if passed:
            c["tag_valid"] = True
            return c

        translation = c["translation"]
        for _ in range(MAX_TAG_RETRIES):
            fix_prompt = TAG_FIX_TEMPLATE.format(
                source=c["source"],
                translation=translation,
                missing=missing,
                extra=extra,
            )
            response = await tag_validator_llm.ainvoke([HumanMessage(content=fix_prompt)])
            content = response.content
            if not isinstance(content, str):
                raise TypeError(
                    f"Expected str content from LLM, got {type(content).__name__}"
                )
            raw = content.strip()
            translation = raw.split("\n")[0].strip().strip('"')
            passed, missing, extra = validate_tags(c["source"], translation)
            if passed:
                break

        if passed:
            logger.info(f"[TAG FIX] {c['source']}: tags corrected")
        else:
            logger.warning(f"[TAG FAIL] {c['source']}: {missing=} {extra=}")

        c["translation"] = translation
        c["tag_valid"] = passed

        return c

    results = await asyncio.gather(*[_validate_one(c) for c in state["candidates"]])
    return {"candidates": list(results)}
