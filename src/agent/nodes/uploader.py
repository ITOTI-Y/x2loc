from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.nodes._helpers import upload_batch
from src.agent.state import AgentState, PatchResult
from src.agent.tools import make_glossary_entry
from src.services.weblate import WeblateClient


def _upload_and_merge(
    batch: Sequence[Mapping[str, Any]],
    value_key: str,
    state: AgentState,
    client: WeblateClient,
    agent_config: AgentConfigSchema,
) -> tuple[list[PatchResult], list[dict], dict]:
    """PATCH a batch of translations and fold the successes into mods_glossary."""
    items = [
        {
            "unit_id": b["unit_id"],
            "source": b["source"],
            "target": b[value_key],
            "context": b["context"],
        }
        for b in batch
    ]
    results, history = upload_batch(items, client=client, agent_config=agent_config)

    ok_ids = {r["unit_id"] for r in results if r["status"] == "ok"}
    mods = dict(state["mods_glossary"])
    for b in batch:
        if b["unit_id"] in ok_ids:
            mods[b["source"]] = make_glossary_entry(
                b["source"], b[value_key], b["context"]
            )
    return results, history, mods


def auto_uploader(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    results, history, mods = _upload_and_merge(
        state["auto_batch"], "translation", state, client, agent_config
    )

    stats = dict(state["stats"])
    stats["auto"] += sum(1 for r in results if r["status"] == "ok")

    score_map = {s["unit_id"]: s["score"] for s in state["scores"]}
    for c in state["auto_batch"]:
        logger.info(
            f"[AUTO] {c['source']} → {c['translation']} ({score_map.get(c['unit_id'], '?')})"
        )

    return {
        "patch_results": results,
        "approved_history": history,
        "stats": stats,
        "mods_glossary": mods,
    }


def review_uploader(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    results, history, mods = _upload_and_merge(
        state["review_approved"], "target", state, client, agent_config
    )

    for a in state["review_approved"]:
        logger.info(f"[REVIEW UPLOAD] {a['source']} → {a['target']}")

    return {
        "patch_results": results,
        "approved_history": history,
        "mods_glossary": mods,
    }
