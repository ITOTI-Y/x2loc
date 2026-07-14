from __future__ import annotations

from src.agent.config import AgentConfigSchema
from src.agent.llm import (
    build_scorer_llm,
    build_tag_validator_llm,
    build_translator_llm,
)
from src.agent.nodes.context_collector import (
    ContextResultsOutputSchema,
    context_collector,
)
from src.agent.nodes.decision_router import decision_router
from src.agent.nodes.fetch_empty import (
    FetchEmptyOutputSchema,
    UnitIterator,
    fetch_empty,
)
from src.agent.nodes.glossary_loader import GlossaryLoaderOutputSchema, glossary_loader
from src.agent.nodes.pattern_extractor import pattern_extractor
from src.agent.nodes.scorer import scorer
from src.agent.nodes.tag_validator import tag_validator
from src.agent.nodes.translator import translator
from src.agent.nodes.uploader import auto_uploader, review_uploader
from src.agent.nodes.user_review import user_review
from src.models.agent import NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient


class WorkflowNodes:
    """Binds client/config (and the once-built LLM runnables) to graph nodes.

    Config is immutable for the graph's lifetime, so the LLM runnables are
    built once here instead of once per node invocation.
    """

    def __init__(
        self,
        client: AsyncWeblateClient,
        config: AgentConfigSchema,
    ) -> None:
        self._client = client
        self._config = config
        self._unit_iterator = UnitIterator(client)
        self._translator_llm = build_translator_llm(config)
        self._tag_validator_llm = build_tag_validator_llm(config)
        self._scorer_llm = build_scorer_llm(config)
        self.user_review = user_review
        self.pattern_extractor = pattern_extractor

    async def glossary_loader(
        self, state: NewAgentStateSchema
    ) -> GlossaryLoaderOutputSchema:
        return await glossary_loader(
            state, client=self._client, agent_config=self._config
        )

    async def fetch_empty(self, state: NewAgentStateSchema) -> FetchEmptyOutputSchema:
        return await fetch_empty(
            state, unit_iterator=self._unit_iterator, agent_config=self._config
        )

    async def context_collector(
        self, state: NewAgentStateSchema
    ) -> ContextResultsOutputSchema:
        return await context_collector(state, client=self._client)

    async def translator(self, state: NewAgentStateSchema) -> NewAgentStateSchema:
        return await translator(
            state, agent_config=self._config, llm=self._translator_llm
        )

    async def tag_validator(self, state: NewAgentStateSchema) -> NewAgentStateSchema:
        return await tag_validator(
            state, agent_config=self._config, llm=self._tag_validator_llm
        )

    async def scorer(self, state: NewAgentStateSchema) -> NewAgentStateSchema:
        return await scorer(state, agent_config=self._config, llm=self._scorer_llm)

    def decision_router(self, state: NewAgentStateSchema) -> NewAgentStateSchema:
        return decision_router(state, agent_config=self._config)

    def auto_uploader(self, state: NewAgentStateSchema) -> NewAgentStateSchema:
        return auto_uploader(state, client=self._client, agent_config=self._config)

    def review_uploader(self, state: NewAgentStateSchema) -> NewAgentStateSchema:
        return review_uploader(state, client=self._client, agent_config=self._config)
