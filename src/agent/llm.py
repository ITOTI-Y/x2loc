from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import Field

from src.agent.config import AgentConfigSchema
from src.models._share import BaseSchema


class DeductionSchema(BaseSchema):
    dim: str
    pts: int
    reason: str


class TranslationOutputSchema(BaseSchema):
    result: str = Field(description="The translated result")


class ScoreOutputSchema(BaseSchema):
    raw_translation: str = Field(description="The raw translation")
    score: int = Field(ge=0, le=100)
    deductions: list[DeductionSchema] = Field(default_factory=list)
    suggested_alternative: str | None = None
    notes: str | None = None


def build_translator_llm(config: AgentConfigSchema) -> Runnable:
    return ChatOpenAI(
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
        model_name=config.translation_model_name,
        temperature=config.translation_temperature,
        max_tokens=2048,
    ).with_structured_output(TranslationOutputSchema)


def build_tag_validator_llm(config: AgentConfigSchema) -> Runnable:
    return ChatOpenAI(
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
        model_name=config.validate_model_name,
        temperature=0.0,
        max_tokens=2048,
    )


def build_scorer_llm(config: AgentConfigSchema) -> Runnable:
    schema = _inline_refs(ScoreOutputSchema.model_json_schema())
    schema.setdefault("title", "ScoreOutputSchema")
    return ChatOpenAI(
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
        model_name=config.scoring_model_name,
        temperature=0.0,
        max_tokens=2048,
    ).with_structured_output(schema)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $ref/$defs so Gemini's response_schema can consume the JSON Schema."""
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
