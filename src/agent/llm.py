from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from src.agent.config import AgentConfigSchema
from src.agent.prompts import (
    scoring_system_blocks,
    tag_fix_system_blocks,
    translation_system_blocks,
)
from src.models.agent import ScoreResultSchema, TranslationOutputSchema


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
        temperature=config.validate_temperature,
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
