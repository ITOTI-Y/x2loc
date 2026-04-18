from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.config import AgentConfigSchema
from src.agent.nodes import WorkflowNodes
from src.agent.state import AgentState
from src.services.weblate import WeblateClient


def route_after_fetch(state: AgentState) -> str:
    return "continue" if state["should_continue"] else "end"


def route_after_decision(state: AgentState) -> str:
    has_auto = len(state["auto_batch"]) > 0
    has_review = state["needs_review"]
    if has_auto and has_review:
        return "both"
    if has_auto:
        return "auto_only"
    return "review_only"


def route_after_auto_upload(state: AgentState) -> str:
    return "review" if state["needs_review"] else "extract"


def build_graph(config: AgentConfigSchema) -> CompiledStateGraph:
    client = WeblateClient(config.weblate)

    nodes = WorkflowNodes(client, config)

    builder = StateGraph(AgentState)  # type: ignore

    builder.add_node("glossary_loader", nodes.glossary_loader)
    builder.add_node("fetch_empty", nodes.fetch_empty)
    builder.add_node("context_collector", nodes.context_collector)
    builder.add_node("translator", nodes.translator)
    builder.add_node("tag_validator", nodes.tag_validator)
    builder.add_node("scorer", nodes.scorer)
    builder.add_node("decision_router", nodes.decision_router)
    builder.add_node("auto_uploader", nodes.auto_uploader)
    builder.add_node("user_review", nodes.user_review)
    builder.add_node("review_uploader", nodes.review_uploader)
    builder.add_node("pattern_extractor", nodes.pattern_extractor)

    builder.add_edge(START, "glossary_loader")
    builder.add_edge("glossary_loader", "fetch_empty")
    builder.add_conditional_edges(
        "fetch_empty", route_after_fetch, {
            "continue": "context_collector", "end": END}
    )
    builder.add_edge("context_collector", "translator")
    builder.add_edge("translator", "tag_validator")
    builder.add_edge("tag_validator", "scorer")
    builder.add_edge("scorer", "decision_router")
    builder.add_conditional_edges(
        "decision_router",
        route_after_decision,
        {
            "auto_only": "auto_uploader",
            "review_only": "user_review",
            "both": "auto_uploader",
        },
    )
    builder.add_conditional_edges(
        "auto_uploader",
        route_after_auto_upload,
        {"review": "user_review", "extract": "pattern_extractor"},
    )
    builder.add_edge("user_review", "review_uploader")
    builder.add_edge("review_uploader", "pattern_extractor")
    builder.add_edge("pattern_extractor", "fetch_empty")

    return builder.compile(checkpointer=InMemorySaver())  # type: ignore
