from __future__ import annotations

from typing import Any, Final

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from src.agent.config import AgentConfigSchema
from src.agent.prompts import (
    scoring_system_blocks,
    tag_fix_system_blocks,
    translation_system_blocks,
)
from src.models._share import BaseSchema
from src.models.agent import ScoreResultSchema, TranslationOutputSchema

# Single source for the suggested_translation description. The static Field
# text below uses the placeholder verbatim; build_scorer_llm re-renders it
# with the configured threshold before handing the schema to the provider.
_SUGGESTED_TRANSLATION_DESC: Final = (
    "The suggested translation if the score is less than {threshold}, "
    "must reply in target language, "
    "if no suggested translation is needed, return null"
)


class DeductionSchema(BaseSchema):
    dim: str = Field(description="The dimension of the deduction")
    pts: int = Field(description="The points of the deduction")
    reason: str = Field(description="The reason for the deduction")


class ScoreOutputSchema(BaseSchema):
    raw_translation: str = Field(description="The raw translation")
    score: int = Field(
        ge=0, le=100, description="The score of the translation between 0 and 100"
    )
    deductions: list[DeductionSchema] = Field(
        ...,
        description="The deductions resulting from the scoring, must reply in target language, if no deduction is needed, return an empty list",
        default_factory=list,
    )
    suggested_translation: str | None = Field(
        None,
        description=_SUGGESTED_TRANSLATION_DESC.format(threshold="the threshold"),
    )
    notes: str | None = Field(
        None,
        description="The notes for the scoring, must reply in target language, if no notes are needed, return null",
    )


def build_translator_llm(config: AgentConfigSchema) -> CompiledStateGraph:
    llm = ChatOpenAI(
        model=config.translation_model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.translation_temperature,
        max_completion_tokens=4096,
    )
    system_blocks = translation_system_blocks(config.target_lang)
    system_message = SystemMessage(content=[dict(system_blocks)])
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=system_message,
        response_format=ToolStrategy(TranslationOutputSchema),
    )


def build_tag_validator_llm(config: AgentConfigSchema) -> CompiledStateGraph:
    system_blocks = tag_fix_system_blocks(config.target_lang)
    system_message = SystemMessage(content=[dict(system_blocks)])
    llm = ChatOpenAI(
        model=config.validate_model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=0.0,
        max_completion_tokens=4096,
    )
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=system_message,
        response_format=ToolStrategy(TranslationOutputSchema),
    )


def build_scorer_llm(config: AgentConfigSchema) -> CompiledStateGraph:
    llm = ChatOpenAI(
        model=config.scoring_model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.scoring_temperature,
        max_completion_tokens=4096,
    )
    system_blocks = scoring_system_blocks(config.target_lang)
    system_message = SystemMessage(content=[dict(system_blocks)])
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=system_message,
        response_format=ToolStrategy(ScoreResultSchema),
    )


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                name = node["$ref"].split("/")[-1]
                return _resolve(defs[name])
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    return _resolve(schema)
