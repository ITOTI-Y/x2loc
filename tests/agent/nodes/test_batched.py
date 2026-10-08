from typing import Any

from src.agent.nodes._batched import invoke_batched
from src.models.agent import TranslationBatchOutputSchema, TranslationItemSchema


class FakeAgent:
    """Answers each request from a script keyed by the request's position."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    async def abatch(self, inputs, config, return_exceptions):
        self.prompts = [i["messages"][0]["content"] for i in inputs]
        return self.replies


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
        max_concurrency=4,
        label="Test",
    )

    assert len(agent.prompts) == 3
    assert agent.prompts[0] == "## Item 1\nprompt 1\n\n## Item 2\nprompt 2"
    assert {i: r.result for i, r in results.items()} == {1: "一", 5: "五"}
