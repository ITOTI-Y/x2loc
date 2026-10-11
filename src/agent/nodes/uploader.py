import asyncio
from typing import NamedTuple, TypedDict

from httpx2 import TransportError
from loguru import logger

from src.models.agent import NewAgentStateSchema, StatsSchema
from src.models.weblate import WeblateUnitPatchSchema
from src.services.weblate import (
    WEBLATE_STATE_FUZZY,
    WEBLATE_STATE_TRANSLATED,
    AsyncWeblateClient,
    WeblateAPIError,
)


class UploaderOutputSchema(TypedDict):
    stats: StatsSchema


class UnitWrite(NamedTuple):
    unit_id: int
    target: str
    state: int
    # Empty for none; otherwise posted on the unit after the target.
    comment: str


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

    def submit(self, items: list[UnitWrite]) -> None:
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

    async def _upload(self, items: list[UnitWrite]) -> None:
        results = await asyncio.gather(*[self._patch(item) for item in items])
        failed = len(items) - sum(results)
        if failed:
            raise WeblateAPIError(502, f"{failed}/{len(items)} unit patches")
        logger.success(f"[UPLOAD] {len(items)} units patched")

    async def _patch(self, item: UnitWrite) -> bool:
        try:
            await self._client.patch_unit(
                item.unit_id,
                WeblateUnitPatchSchema(target=[item.target], state=item.state),
            )
            # A unit handed to a human without its reason is a failed write.
            if item.comment:
                await self._client.add_unit_comment(item.unit_id, item.comment)
        except (WeblateAPIError, TransportError) as exc:
            logger.error(f"Write failed for unit {item.unit_id}: {exc!r}")
            return False
        return True


async def uploader(
    state: NewAgentStateSchema, *, background_uploader: BackgroundUploader
) -> UploaderOutputSchema:
    units = {u.id: u for u in state.scores}
    items: list[UnitWrite] = []
    n_approved = n_modified = n_skipped = n_needs_editing = 0

    for decision in state.decisions:
        unit = units.get(decision.unit_id)
        fallback = unit.translated if unit is not None else ""
        target = decision.translation or fallback
        if decision.action == "skip" or not target:
            n_skipped += 1
            continue
        state_code = WEBLATE_STATE_TRANSLATED
        if decision.action == "approve":
            n_approved += 1
        elif decision.action == "needs_editing":
            n_needs_editing += 1
            state_code = WEBLATE_STATE_FUZZY
        else:
            n_modified += 1
        items.append(UnitWrite(decision.unit_id, target, state_code, decision.comment))

    background_uploader.submit(items)

    return {
        "stats": StatsSchema(
            auto=state.stats["auto"],
            approved=state.stats["approved"] + n_approved,
            modified=state.stats["modified"] + n_modified,
            skipped=state.stats["skipped"] + n_skipped,
            needs_editing=state.stats["needs_editing"] + n_needs_editing,
        )
    }
