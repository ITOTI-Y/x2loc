from __future__ import annotations

from os.path import commonprefix

from loguru import logger

from src.agent.state import PatchResult
from src.services.weblate import WeblateClient


def upload_batch(
    candidates: list[dict],
    *,
    client: WeblateClient,
) -> tuple[list[PatchResult], list[dict]]:
    """PATCH translations to Weblate.

    Tag validity is guaranteed upstream: tag-invalid candidates are scored 0
    and routed to review, and user_review re-validates edited text before
    approving — so no re-check happens here.
    """
    results: list[PatchResult] = []
    history: list[dict] = []

    for c in candidates:
        source = c.get("source", "")
        translation = c.get("target", c.get("translation", ""))

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
    # os.path.commonprefix is character-level despite the name, not path-aware.
    return commonprefix([a, b])


def common_suffix(a: str, b: str) -> str:
    if not a or not b:
        return ""
    i = 0
    for ca, cb in zip(reversed(a), reversed(b), strict=False):
        if ca != cb:
            break
        i += 1
    return a[len(a) - i :] if i > 0 else ""
