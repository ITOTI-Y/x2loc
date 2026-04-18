from __future__ import annotations

from loguru import logger

from src.agent.config import AgentConfigSchema
from src.agent.state import PatchResult
from src.agent.tools import validate_tags
from src.services.weblate import WeblateClient


def upload_batch(
    candidates: list[dict],
    *,
    client: WeblateClient,
    agent_config: AgentConfigSchema,
) -> tuple[list[PatchResult], list[dict]]:
    results: list[PatchResult] = []
    history: list[dict] = []

    for c in candidates:
        source = c.get("source", "")
        translation = c.get("target", c.get("translation", ""))

        if agent_config.dry_run:
            results.append({"unit_id": c["unit_id"], "status": "ok", "error": None})
            history.append({"source": source, "target": translation})
            continue

        if source:
            passed, _, _ = validate_tags(source, translation)
            if not passed:
                results.append(
                    {
                        "unit_id": c["unit_id"],
                        "status": "tag_fail",
                        "error": "Pre-flight tag validation failed",
                    }
                )
                continue

        try:
            client.patch_unit(c["unit_id"], {"target": [translation], "state": 20})
            results.append({"unit_id": c["unit_id"], "status": "ok", "error": None})
            history.append({"source": source, "target": translation})
        except Exception as e:
            logger.error(f"PATCH failed for unit {c['unit_id']}: {e}")
            results.append(
                {"unit_id": c["unit_id"], "status": "error", "error": str(e)}
            )

    return results, history


def common_prefix(a: str, b: str) -> str:
    i = 0
    for ca, cb in zip(a, b, strict=True):
        if ca != cb:
            break
        i += 1
    return a[:i]


def common_suffix(a: str, b: str) -> str:
    if not a or not b:
        return ""
    i = 0
    for ca, cb in zip(reversed(a), reversed(b), strict=True):
        if ca != cb:
            break
        i += 1
    return a[len(a) - i :] if i > 0 else ""
