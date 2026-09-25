import httpx

from src.agent.config import ConfigSchema
from src.agent.llm import _chat_model

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "译文"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


async def test_transient_status_is_retried_at_request_level(
    agent_config: ConfigSchema,
) -> None:
    statuses = iter([520, 429, 200])
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        calls.append(status)
        if status != 200:
            return httpx.Response(status, headers={"retry-after": "0.01"}, json={})
        return httpx.Response(200, json=COMPLETION)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = _chat_model(
        model="test-model",
        config=agent_config,
        temperature=0.0,
        http_async_client=client,
    )
    reply = await model.ainvoke("hello")
    await client.aclose()
    assert reply.content == "译文"
    assert calls == [520, 429, 200]
