from __future__ import annotations

from typing import TypedDict

from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.llm import TranslationOutputSchema
from src.agent.prompts import format_translation_prompt
from src.agent.tools import lookup_glossary_or_patterns
from src.models.agent import (
    ComponentInfoSchema,
    NewAgentStateSchema,
    TranslationUnitSchema,
    WeblateUnitSchema,
)


class TranslateOutputSchema(TypedDict):
    candidates: list[TranslationUnitSchema]


async def translator(
    state: NewAgentStateSchema,
    *,
    agent_config: AgentConfigSchema,
    agent: CompiledStateGraph,
) -> TranslateOutputSchema:

    def _build_translate_input(
        unit: WeblateUnitSchema, ctx: list[ComponentInfoSchema]
    ) -> dict:
        base_matches = lookup_glossary_or_patterns(unit.source, state.base_glossary)
        mods_matches = lookup_glossary_or_patterns(unit.source, state.mods_glossary)
        match_patterns = lookup_glossary_or_patterns(unit.source, state.patterns)
        prompt = format_translation_prompt(
            unit.source,
            unit.note,
            base_matches,
            mods_matches,
            ctx,
            match_patterns,
        )
        return {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }

    inputs = [
        _build_translate_input(u, state.context_results[u.id])
        for u in state.to_translate
    ]
    responses = await agent.abatch(inputs, config={"max_concurrency": 10})

    candidates = []
    for u, r in zip(state.to_translate, responses, strict=True):
        if r is not None:
            structured = r.get("structured_response")
            if isinstance(structured, TranslationOutputSchema):
                result_text = structured.result
                candidates.append(
                    TranslationUnitSchema(
                        id=u.id,
                        source=u.source,
                        translated=result_text,
                        key=u.context,
                        context=state.context_results[u.id],
                        category=u.note or "unknown",
                        pattern_matched=len(
                            lookup_glossary_or_patterns(u.source, state.patterns)
                        )
                        > 0,
                        glossary_base=lookup_glossary_or_patterns(
                            u.source, state.base_glossary
                        ),
                        glossary_mods=lookup_glossary_or_patterns(
                            u.source, state.mods_glossary
                        ),
                        tag_valid=False,
                        original_unit=u,
                        patterns=lookup_glossary_or_patterns(u.source, state.patterns),
                    )
                )
            else:
                logger.warning(f"Unexpected response type: {type(structured).__name__}")
        else:
            candidates.append(
                TranslationUnitSchema(
                    id=u.id,
                    source=u.source,
                    translated="",
                    key=u.context,
                    context=state.context_results[u.id],
                    category=u.note or "unknown",
                    pattern_matched=len(
                        lookup_glossary_or_patterns(u.source, state.patterns)
                    )
                    > 0,
                    glossary_base=lookup_glossary_or_patterns(
                        u.source, state.base_glossary
                    ),
                    glossary_mods=lookup_glossary_or_patterns(
                        u.source, state.mods_glossary
                    ),
                    tag_valid=False,
                    original_unit=u,
                    patterns=lookup_glossary_or_patterns(u.source, state.patterns),
                )
            )
    logger.success(f"Translated {len(candidates)} units")
    return {"candidates": candidates}
