from __future__ import annotations

from langgraph.types import interrupt
from loguru import logger

from src.agent.state import AgentState
from src.agent.tools import validate_tags


def user_review(state: AgentState) -> dict:
    score_map = {s["unit_id"]: s for s in state["scores"]}
    review_items = [
        {
            "unit_id": c["unit_id"],
            "source": c["source"],
            "translation": c["translation"],
            "category": c["category"],
            "score": score_map.get(c["unit_id"], {}).get("score", 0),
            "deductions": score_map.get(c["unit_id"], {}).get("deductions", []),
            "suggested_alternative": score_map.get(c["unit_id"], {}).get(
                "suggested_alternative"
            ),
        }
        for c in state["review_batch"]
    ]

    decisions = interrupt(review_items)

    approved: list[dict] = []
    skip_ids = list(state["skip_ids"])
    stats = dict(state["stats"])

    for d in decisions:
        action = d.get("action", "skip")
        uid = d["unit_id"]

        if action == "skip":
            skip_ids.append(uid)
            stats["skipped"] += 1
            continue

        original = next((c for c in state["review_batch"] if c["unit_id"] == uid), None)
        if not original:
            continue

        if action == "modify":
            translation = d.get("translation", "")
            passed, _, _ = validate_tags(original["source"], translation)
            if not passed:
                logger.warning(
                    f"[REVIEW] Modified translation has tag errors, skipping: {uid}"
                )
                skip_ids.append(uid)
                stats["skipped"] += 1
                continue
            stats["modified"] += 1
        else:
            translation = original["translation"]
            stats["approved"] += 1

        approved.append(
            {
                "unit_id": uid,
                "source": original["source"],
                "target": translation,
                "context": original["context"],
            }
        )

    return {"review_approved": approved, "skip_ids": skip_ids, "stats": stats}
