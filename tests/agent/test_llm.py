import httpx
from pydantic import SecretStr

from src.agent.config import ConfigSchema
from src.agent.llm import _chat_model
from src.models.weblate import WeblateConfigSchema
from src.models.workshop import SteamConfigSchema

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


def _config() -> ConfigSchema:
    return ConfigSchema(
        weblate=WeblateConfigSchema(url="http://weblate", token="t", project_slug="p"),
        steam=SteamConfigSchema(steam_username="u", steam_password=SecretStr("p")),
        translation_model_name="test-model",
        validate_model_name="",
        scoring_model_name="",
        base_url="http://llm.test/v1",
        api_key=SecretStr("k"),
        batch_size=10,
        auto_approve_threshold=95,
        max_concurrency=1,
        base_glossary_slug="b",
        mods_glossary_slug="m",
        custom_glossary_slug="c",
        target_lang="zh_Hans",
    )


async def test_transient_status_is_retried_at_request_level() -> None:
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
        config=_config(),
        temperature=0.0,
        http_async_client=client,
    )
    reply = await model.ainvoke("hello")
    await client.aclose()
    assert reply.content == "译文"
    assert calls == [520, 429, 200]
