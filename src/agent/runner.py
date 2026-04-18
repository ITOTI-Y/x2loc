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

app = typer.Typer(name="agent", help="LangGraph glossary translation agent.")


def _initial_state() -> dict:
    return {
        "base_glossary": {},
        "mods_glossary": {},
        "current_page": 1,
        "remaining_count": -1,
        "current_units": [],
        "context_results": [],
        "candidates": [],
        "scores": [],
        "auto_batch": [],
        "review_batch": [],
        "review_approved": [],
        "patch_results": [],
        "approved_history": [],
        "skip_ids": [],
        "stats": {"auto": 0, "approved": 0, "modified": 0, "skipped": 0},
        "session_patterns": load_cached_patterns(),
        "should_continue": True,
        "needs_review": False,
    }


def _prompt_user_review(items: list[dict]) -> list[dict]:
    decisions: list[dict] = []
    for idx, item in enumerate(items, 1):
        print(f"\n{'━' * 50}")
        print(f"#{idx} [{item['score']}分] {item['source']} → {item['translation']}")
        if item.get("deductions"):
            reasons = ", ".join(
                f"{d['dim']} {d['pts']} ({d['reason']})" for d in item["deductions"]
            )
            print(f"  扣分: {reasons}")

        suggested = item.get("suggested_alternative")
        if suggested:
            print(f"  建议: {suggested}")
            print(f'  [1] 改为 "{suggested}" (推荐)  [2] 通过原版  [3] 跳过')
        else:
            print("  [1] 通过  [2] 跳过")

        choice = input("  选择: ").strip()

        if suggested:
            if choice == "1":
                decisions.append(
                    {
                        "unit_id": item["unit_id"],
                        "action": "modify",
                        "translation": suggested,
                    }
                )
            elif choice == "2":
                decisions.append(
                    {
                        "unit_id": item["unit_id"],
                        "action": "approve",
                        "translation": item["translation"],
                    }
                )
            else:
                decisions.append({"unit_id": item["unit_id"], "action": "skip"})
        else:
            if choice == "1":
                decisions.append(
                    {
                        "unit_id": item["unit_id"],
                        "action": "approve",
                        "translation": item["translation"],
                    }
                )
            else:
                decisions.append({"unit_id": item["unit_id"], "action": "skip"})
    return decisions


def _print_summary(stats: dict[str, int], remaining: int) -> None:
    print(f"\n{'━' * 30}")
    print(f"  Auto (≥95):   {stats.get('auto', 0)}")
    print(f"  Approved:     {stats.get('approved', 0)}")
    print(f"  Modified:     {stats.get('modified', 0)}")
    print(f"  Skipped:      {stats.get('skipped', 0)}")
    print(f"  Remaining:    {remaining}")
    print(f"{'━' * 30}")


async def _run_async(config: AgentConfigSchema) -> None:
    """Drive the graph through its async API."""
    graph = build_graph(config)
    thread: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    stream_input: dict | Command = _initial_state()

    while True:
        async for event in graph.astream(stream_input, thread, stream_mode="updates"):
            for node_name in event:
                if node_name == "glossary_loader":
                    base_n = len(event[node_name].get("base_glossary", {}))
                    mods_n = len(event[node_name].get("mods_glossary", {}))
                    logger.info(f"Glossaries loaded: {base_n} base + {mods_n} mods")

        snapshot = await graph.aget_state(thread)
        if not snapshot.next:
            final = snapshot.values
            _print_summary(final.get("stats", {}), final.get("remaining_count", 0))
            break

        interrupt_value = None
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_value = task.interrupts[0].value
                break

        if interrupt_value is None:
            logger.warning("Graph paused without interrupt value, exiting")
            break

        decisions = await asyncio.to_thread(_prompt_user_review, interrupt_value)
        stream_input = Command(resume=decisions)


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Path to the config file.")
    ],
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Dry run.")] = False,
) -> None:
    """Run the glossary translation agent."""
    config = load_config(str(config_path))
    config = config.model_copy(update={"batch_size": batch_size, "dry_run": dry_run})
    asyncio.run(_run_async(config))


if __name__ == "__main__":
    app()
