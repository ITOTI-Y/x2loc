import asyncio
from typing import TypedDict

from httpx2 import TransportError
from loguru import logger

from src.models.agent import NewAgentStateSchema, StatsSchema
from src.models.weblate import WeblateUnitPatchSchema
from src.services.weblate import (
    WEBLATE_STATE_TRANSLATED,
    AsyncWeblateClient,
    WeblateAPIError,
)


class UploaderOutputSchema(TypedDict):
    stats: StatsSchema


class BackgroundUploader:
    """Fire-and-drain Weblate writes.

    The interactive path submits a batch and moves on to the next one while
    it uploads. Tasks stay referenced until `drain`, which re-raises every
    background failure as one ExceptionGroup — a log line alone would let
    the caller treat a lost batch as success.
    """

    def __init__(self, client: AsyncWeblateClient) -> None:
        self._client = client
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, items: list[tuple[int, str]]) -> None:
        if not items:
            return
        self._tasks.add(asyncio.create_task(self._upload(items)))

    async def drain(self) -> None:
        tasks, self._tasks = self._tasks, set()
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("background Weblate uploads failed", errors)

    async def _upload(self, items: list[tuple[int, str]]) -> None:
        results = await asyncio.gather(
            *[self._patch(unit_id, target) for unit_id, target in items]
        )
        failed = len(items) - sum(results)
        if failed:
            raise WeblateAPIError(502, f"{failed}/{len(items)} unit patches")
        logger.success(f"[UPLOAD] {len(items)} units patched")

    async def _patch(self, unit_id: int, target: str) -> bool:
        try:
            await self._client.patch_unit(
                unit_id,
                WeblateUnitPatchSchema(target=[target], state=WEBLATE_STATE_TRANSLATED),
            )
        except (WeblateAPIError, TransportError) as exc:
            logger.error(f"PATCH failed for unit {unit_id}: {exc!r}")
            return False
        return True


async def uploader(
    state: NewAgentStateSchema, *, background_uploader: BackgroundUploader
) -> UploaderOutputSchema:
    units = {u.id: u for u in state.scores}
    items: list[tuple[int, str]] = []
    n_approved = n_modified = n_skipped = 0

    for decision in state.decisions:
        unit = units.get(decision.unit_id)
        fallback = unit.translated if unit is not None else ""
        target = decision.translation or fallback
        if decision.action == "skip" or not target:
            n_skipped += 1
            continue
        if decision.action == "approve":
            n_approved += 1
        else:
            n_modified += 1
        items.append((decision.unit_id, target))

    background_uploader.submit(items)

    return {
        "stats": StatsSchema(
            auto=state.stats["auto"],
            approved=state.stats["approved"] + n_approved,
            modified=state.stats["modified"] + n_modified,
            skipped=state.stats["skipped"] + n_skipped,
        )
    }
