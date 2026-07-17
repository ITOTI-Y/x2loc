from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.config import AgentConfigSchema
from src.agent.nodes import WorkflowNodes
from src.models.agent import NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient


def route_after_fetch(state: NewAgentStateSchema) -> str:
    return "end" if state.is_end else "continue"


def build_graph(config: AgentConfigSchema) -> CompiledStateGraph:
    client = AsyncWeblateClient(config.weblate)

    nodes = WorkflowNodes(client, config)

    builder = StateGraph(NewAgentStateSchema)

    builder.add_node("glossary_loader", nodes.glossary_loader)
    builder.add_node("fetch_empty", nodes.fetch_empty)
    builder.add_node("context_collector", nodes.context_collector)
    builder.add_node("translator", nodes.translator)
    builder.add_node("tag_validator", nodes.tag_validator)
    builder.add_node("scorer", nodes.scorer)
    builder.add_node("user_review", nodes.user_review)
    builder.add_node("auto_uploader", nodes.auto_uploader)
    builder.add_node("review_uploader", nodes.review_uploader)
    builder.add_node("pattern_extractor", nodes.pattern_extractor)

    builder.add_edge(START, "glossary_loader")
    builder.add_edge("glossary_loader", "fetch_empty")
    builder.add_conditional_edges(
        "fetch_empty", route_after_fetch, {"continue": "context_collector", "end": END}
    )
    builder.add_edge("context_collector", "translator")
    builder.add_edge("translator", "tag_validator")
    builder.add_edge("tag_validator", "scorer")
    builder.add_edge("scorer", "user_review")
    builder.add_edge("auto_uploader", "pattern_extractor")
    builder.add_edge("review_uploader", "pattern_extractor")
    builder.add_edge("pattern_extractor", "fetch_empty")

    return builder.compile(checkpointer=InMemorySaver())
