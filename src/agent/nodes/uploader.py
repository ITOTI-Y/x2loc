from __future__ import annotations

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.nodes._helpers import upload_batch
from src.agent.state import AgentState
from src.services.weblate import WeblateClient


def auto_uploader(
    state: AgentState, *, client: WeblateClient, agent_config: AgentConfigSchema
) -> dict:
    items = [
        {
            "unit_id": c["unit_id"],
            "source": c["source"],
            "target": c["translation"],
            "context": c["context"],
        }
        for c in state["auto_batch"]
    ]
    results, history = upload_batch(items, client=client, agent_config=agent_config)

    stats = dict(state["stats"])
    stats["auto"] += sum(1 for r in results if r["status"] == "ok")

    mods = dict(state["mods_glossary"])
    for c in state["auto_batch"]:
        mods[c["source"]] = {"target": c["translation"], "context": c["context"]}

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
    items = [
        {
            "unit_id": a["unit_id"],
            "source": a["source"],
            "target": a["target"],
            "context": a["context"],
        }
        for a in state["review_approved"]
    ]
    results, history = upload_batch(items, client=client, agent_config=agent_config)

    mods = dict(state["mods_glossary"])
    for a in state["review_approved"]:
        mods[a["source"]] = {"target": a["target"], "context": a["context"]}

    for a in state["review_approved"]:
        logger.info(f"[REVIEW UPLOAD] {a['source']} → {a['target']}")

    return {
        "patch_results": results,
        "approved_history": history,
        "mods_glossary": mods,
    }
