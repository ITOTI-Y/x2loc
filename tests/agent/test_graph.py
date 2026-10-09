from src.agent.config import ConfigSchema
from src.agent.graph import build_graph, route_after_fetch
from src.agent.review import ThresholdReview
from src.models.agent import NewAgentStateSchema
from src.models.weblate import WeblateUnitSchema
from src.services.weblate import AsyncWeblateClient


class _RecordingGlossaries:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def units(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        self.calls.append(slug)
        return []

    async def aclose(self) -> None:
        pass


class _FullyTranslatedClient(AsyncWeblateClient):
    async def list_units(
        self, component_slug: str, language: str, q: str = ""
    ) -> list[WeblateUnitSchema]:
        return []


def test_route_after_fetch_loads_glossaries_once() -> None:
    assert route_after_fetch(NewAgentStateSchema(is_end=True)) == "end"
    assert route_after_fetch(NewAgentStateSchema()) == "load"
    assert route_after_fetch(NewAgentStateSchema(glossaries_loaded=True)) == "continue"


async def test_fully_translated_component_skips_glossary_load(
    agent_config: ConfigSchema,
) -> None:
    client = _FullyTranslatedClient(agent_config.weblate)
    glossaries = _RecordingGlossaries()
    graph, nodes = build_graph(
        agent_config, review=ThresholdReview(), client=client, glossaries=glossaries
    )
    try:
        final = await graph.ainvoke(
            NewAgentStateSchema(component_slug="done"),
            config={"configurable": {"thread_id": "t"}},
        )
    finally:
        await nodes.aclose()
        await client.close()
    assert final["is_end"]
    assert glossaries.calls == []
