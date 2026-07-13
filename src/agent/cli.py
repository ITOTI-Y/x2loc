from __future__ import annotations

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
from src.models.agent import NewAgentStateSchema

app = typer.Typer(name="agent", help="LangGraph glossary translation agent.")


def _decide(item: dict, choice: str) -> dict:
    """Map one user input to a review decision.

    Numbered choices resolve via the option table; any other non-empty
    input is a hand-typed translation and must be recorded as "modify" —
    user_review only honors the custom text for that action; empty input
    skips.
    """
    suggested = item.get("suggested_translation")
    if suggested:
        options = {
            "1": ("modify", suggested),
            "2": ("approve", item["translation"]),
            "3": ("skip", None),
        }
    else:
        options = {
            "1": ("approve", item["translation"]),
            "2": ("skip", None),
        }

    if choice in options:
        action, translation = options[choice]
    elif choice:
        action, translation = "modify", choice
    else:
        action, translation = "skip", None

    decision: dict = {"unit_id": item["unit_id"], "action": action}
    if translation is not None:
        decision["translation"] = translation
    return decision


def _prompt_user_review(items: list[dict], auto_skip: bool = False) -> list[dict]:
    decisions: list[dict] = []
    for idx, item in enumerate(items, 1):
        print(f"\n{'━' * 50}")
        print(f"#{idx} [{item['score']}分] {item['source']} → {item['translation']}")
        if item.get("deductions"):
            reasons = ", ".join(
                f"{d['dim']} {d['pts']} ({d['reason']})" for d in item["deductions"]
            )
            print(f"  Deductions: {reasons}")

        suggested = item.get("suggested_translation")
        if suggested:
            print(f"  Suggested: {suggested}")
            print(
                f'  [1] Modify to "{suggested}" (recommended)  [2] Accept original  [3] Skip'
            )
        else:
            print("  [1] Accept  [2] Skip")

        # Empty input maps to "skip" regardless of the option table, so
        # auto-skip never depends on a numbered choice existing.
        choice = "" if auto_skip else input("  Choice or Translation: ").strip()
        decisions.append(_decide(item, choice))
    return decisions


def _print_summary(stats: dict[str, int], remaining: int) -> None:
    logger.info(
        f"Batch complete: auto={stats.get('auto', 0)} approved={stats.get('approved', 0)} "
        f"modified={stats.get('modified', 0)} skipped={stats.get('skipped', 0)} "
        f"remaining={remaining}"
    )

async def _run_async(config: AgentConfigSchema, auto_skip: bool) -> None:
    """Drive the graph through its async API."""
    graph = build_graph(config)
    thread: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    state: NewAgentStateSchema | Command = NewAgentStateSchema(patterns=load_cached_patterns())

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

        interrupts = final.get("__interrupts__")
        if not interrupts:
            breakpoint()
            break

        decisions = await asyncio.to_thread(
            _prompt_user_review, interrupts, auto_skip=auto_skip
        )
        state = Command(resume=decisions)


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Path to the config file.")
    ],
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Dry run.")] = False,
    auto_skip: Annotated[bool, typer.Option("--auto-skip", help="Auto skip.")] = False,
) -> None:
    """Run the glossary translation agent."""
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be a positive integer")
    config = load_config(str(config_path))
    config = config.model_copy(update={"batch_size": batch_size, "dry_run": dry_run})
    asyncio.run(_run_async(config, auto_skip=auto_skip))


if __name__ == "__main__":
    app()
