from langgraph.types import interrupt
from loguru import logger

from src.agent.tools import validate_tags
from src.models.agent import NewAgentStateSchema, ReviewDecisionSchema


def user_review(state: NewAgentStateSchema) -> dict:
    scores = state.scores

    decisions: list[ReviewDecisionSchema] = interrupt(scores)

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
        else:
            translation = original["translation"]

        passed, _, _ = validate_tags(original["source"], translation)
        if not passed:
            logger.warning(
                f"[REVIEW] {action.capitalize()} translation has tag errors, "
                f"skipping: {uid}"
            )
            skip_ids.append(uid)
            stats["skipped"] += 1
            continue

        if action == "modify":
            stats["modified"] += 1
        else:
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
