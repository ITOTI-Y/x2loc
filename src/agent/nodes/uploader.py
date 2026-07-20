import asyncio
from typing import Final, TypedDict

from loguru import logger

from src.models.agent import NewAgentStateSchema, StatsSchema
from src.services.weblate import AsyncWeblateClient

WEBLATE_STATE_TRANSLATED: Final = 20


class UploaderOutputSchema(TypedDict):
    stats: StatsSchema


class BackgroundUploader:
    def __init__(self, client: AsyncWeblateClient) -> None:
        self._client = client
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, items: list[tuple[int, str]]) -> None:
        if not items:
            return
        task = asyncio.create_task(self._upload(items))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks))

    async def _upload(self, items: list[tuple[int, str]]) -> None:
        results = await asyncio.gather(
            *[self._patch(unit_id, target) for unit_id, target in items]
        )
        logger.success(f"[UPLOAD] {sum(results)}/{len(items)} units patched")

    async def _patch(self, unit_id: int, target: str) -> bool:
        try:
            await self._client.patch_unit(
                unit_id, {"target": [target], "state": WEBLATE_STATE_TRANSLATED}
            )
        except Exception as e:
            logger.error(f"PATCH failed for unit {unit_id}: {e!r}")
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
        if unit is None:
            continue
        target = decision.translation or unit.translated
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
