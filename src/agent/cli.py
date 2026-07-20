import asyncio
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from loguru import logger

from src.agent.config import AgentConfigSchema, load_config
from src.agent.graph import build_graph
from src.agent.nodes.pattern_extractor import load_cached_patterns
from src.models.agent import NewAgentStateSchema, TranslationUnitSchema
from src.ui.user import prompt_user_review

app = typer.Typer(name="agent", help="LangGraph glossary translation agent.")


def _print_summary(stats: dict[str, int], remaining: int) -> None:
    logger.info(
        f"Batch complete: auto={stats.get('auto', 0)} approved={stats.get('approved', 0)} "
        f"modified={stats.get('modified', 0)} skipped={stats.get('skipped', 0)} "
        f"remaining={remaining}"
    )


async def _run_async(config: AgentConfigSchema, auto_accept: bool) -> None:
    """Drive the graph through its async API."""
    graph = build_graph(config)
    thread: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    state: NewAgentStateSchema | Command = NewAgentStateSchema(
        patterns=load_cached_patterns()
    )

    while True:
        final = None
        async for event in graph.astream(
            input=state, config=thread, stream_mode="updates"
        ):
            final = event
            pass

        if final is None:
            breakpoint()
            break

        interrupt = final.get(("__interrupt__"), {})
        if not interrupt:
            breakpoint()
            break

        scores: list[TranslationUnitSchema] = interrupt[0].value

        decisions = await asyncio.to_thread(
            prompt_user_review, scores, auto_accept=auto_accept
        )
        state = Command(resume=decisions)


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Path to the config file.")
    ],
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size.")],
    auto_accept: Annotated[
        bool, typer.Option("--auto-accept", help="Auto accept.")
    ] = False,
) -> None:
    """Run the glossary translation agent."""
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be a positive integer")
    config = load_config(str(config_path))
    config = config.model_copy(update={"batch_size": batch_size})
    asyncio.run(_run_async(config, auto_accept=auto_accept))


if __name__ == "__main__":
    app()
