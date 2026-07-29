from __future__ import annotations

import math

from httpx import AsyncClient, Client
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.nodes import WorkflowNodes
from src.agent.review import ReviewPolicy
from src.models.agent import NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient

serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("src.models.weblate", "WeblateUnitSchema"),
        ("src.models.agent", "PatternSchema"),
        ("src.models.agent", "ComponentInfoSchema"),
        ("src.models.agent", "TranslationUnitSchema"),
    ]
)


def route_after_fetch(state: NewAgentStateSchema) -> str:
    return "end" if state.is_end else "continue"


def route_after_review(state: NewAgentStateSchema) -> str:
    return "retry" if state.retry_pending else "done"


def graph_recursion_limit(
    unit_count: int, *, batch_size: int, max_attempts: int
) -> int:
    """Upper bound on super-steps for one automatic-path invocation.

    Each batch spends ~7 steps on the happy path and 4 more per quality
    retry round; the constant headroom covers glossary loading, the final
    empty fetch and conditional-edge bookkeeping. A fixed cap cannot work:
    a 26K-unit component at batch_size=10 needs thousands of steps.
    """
    batches = math.ceil(max(1, unit_count) / batch_size)
    return 16 + batches * (8 + 4 * (max_attempts - 1))


def build_graph(
    config: AgentConfigSchema,
    *,
    review: ReviewPolicy,
    client: AsyncWeblateClient | None = None,
    http_client: Client | None = None,
    http_async_client: AsyncClient | None = None,
) -> tuple[CompiledStateGraph, WorkflowNodes]:
    """Compile the translation graph around one review policy.

    Passing `client` lets a long-lived service share one Weblate connection
    pool across every component, and the two `httpx` clients do the same for
    LLM traffic; the interactive CLI omits them and the graph owns its
    resources for its lifetime.
    """
    owns_client = client is None
    nodes = WorkflowNodes(
        client or AsyncWeblateClient(config.weblate),
        config,
        review=review,
        owns_client=owns_client,
        http_client=http_client,
        http_async_client=http_async_client,
    )

    builder = StateGraph(NewAgentStateSchema)

    builder.add_node("glossary_loader", nodes.glossary_loader)
    builder.add_node("fetch_empty", nodes.fetch_empty)
    builder.add_node("context_collector", nodes.context_collector)
    builder.add_node("translator", nodes.translator)
    builder.add_node("tag_validator", nodes.tag_validator)
    builder.add_node("scorer", nodes.scorer)
    builder.add_node("review", nodes.review)
    builder.add_node("uploader", nodes.uploader)

    builder.add_edge(START, "glossary_loader")
    builder.add_edge("glossary_loader", "fetch_empty")
    builder.add_conditional_edges(
        "fetch_empty", route_after_fetch, {"continue": "context_collector", "end": END}
    )
    builder.add_edge("context_collector", "translator")
    builder.add_edge("translator", "tag_validator")
    builder.add_edge("tag_validator", "scorer")
    builder.add_edge("scorer", "review")
    builder.add_conditional_edges(
        "review", route_after_review, {"retry": "translator", "done": "uploader"}
    )

    if review.extracts_patterns:
        builder.add_node("pattern_extractor", nodes.pattern_extractor)
        builder.add_edge("uploader", "pattern_extractor")
        builder.add_edge("pattern_extractor", "fetch_empty")
    else:
        builder.add_edge("uploader", "fetch_empty")

    logger.debug("Compiled graph with review policy {}", type(review).__name__)
    return builder.compile(checkpointer=InMemorySaver(serde=serde)), nodes
