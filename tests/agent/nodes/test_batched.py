import asyncio
from typing import Any

from src.agent.nodes._batched import invoke_batched
from src.models.agent import TranslationBatchOutputSchema, TranslationItemSchema


class FakeAgent:
    """Answers each request from a script keyed by the request's prompt order."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    async def ainvoke(self, request):
        self.prompts.append(request["messages"][0]["content"])
        reply = self.replies[len(self.prompts) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return reply


def batch(*pairs: tuple[int, str]) -> dict[str, Any]:
    items = [TranslationItemSchema(id=i, result=r) for i, r in pairs]
    return {"structured_response": TranslationBatchOutputSchema(results=items)}


async def test_invoke_batched_chunks_and_maps_results_by_id() -> None:
    agent = FakeAgent(
        [
            batch((1, "一"), (99, "not asked")),
            RuntimeError("timeout"),
            batch((5, "五")),
        ]
    )
    items = [(i, f"prompt {i}") for i in range(1, 6)]

    results = await invoke_batched(
        agent,  # ty: ignore[invalid-argument-type]
        items,
        units_per_request=2,
        slots=asyncio.Semaphore(4),
        label="Test",
    )

    assert len(agent.prompts) == 3
    assert agent.prompts[0] == "## Item 1\nprompt 1\n\n## Item 2\nprompt 2"
    assert {i: r.result for i, r in results.items()} == {1: "一", 5: "五"}


async def test_invoke_batched_keeps_requests_in_flight_within_slots() -> None:
    in_flight = peak = 0

    class SlowAgent:
        async def ainvoke(self, request):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return batch()

    slots = asyncio.Semaphore(2)
    await asyncio.gather(
        *(
            invoke_batched(
                SlowAgent(),  # ty: ignore[invalid-argument-type]
                [(i, "p") for i in range(start, start + 6)],
                units_per_request=1,
                slots=slots,
                label="Test",
            )
            for start in (0, 100)
        )
    )

    assert peak == 2
