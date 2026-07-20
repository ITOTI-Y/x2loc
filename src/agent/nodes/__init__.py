import asyncio

from loguru import logger

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
from src.agent.nodes.fetch_empty import (
    FetchEmptyOutputSchema,
    UnitIterator,
    fetch_empty,
)
from src.agent.nodes.glossary_loader import GlossaryLoaderOutputSchema, glossary_loader
from src.agent.nodes.pattern_extractor import pattern_extractor
from src.agent.nodes.scorer import ScorerOutputSchema, scorer
from src.agent.nodes.tag_validator import TagValidatorOutputSchema, tag_validator
from src.agent.nodes.translator import TranslateOutputSchema, translator
from src.agent.nodes.uploader import BackgroundUploader, UploaderOutputSchema, uploader
from src.agent.nodes.user_review import user_review
from src.models.agent import ComponentInfoSchema, NewAgentStateSchema
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
        self._translator_agent = build_translator_llm(config)
        self._tag_validator_llm = build_tag_validator_llm(config)
        self._scorer_llm = build_scorer_llm(config)
        self._prefetch_task: (
            asyncio.Task[dict[int, list[ComponentInfoSchema]]] | None
        ) = None
        self._background_uploader = BackgroundUploader(client)
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
        prefetched = await self._harvest_prefetch()
        hit = {u.id: prefetched[u.id] for u in state.to_translate if u.id in prefetched}
        missing = [u for u in state.to_translate if u.id not in prefetched]
        fresh = await context_collector(missing, client=self._client)
        self._start_prefetch()
        return {
            "context_results": {**hit, **fresh},
        }

    async def translator(self, state: NewAgentStateSchema) -> TranslateOutputSchema:
        return await translator(
            state, agent_config=self._config, agent=self._translator_agent
        )

    async def tag_validator(
        self, state: NewAgentStateSchema
    ) -> TagValidatorOutputSchema:
        return await tag_validator(
            state, agent_config=self._config, llm=self._tag_validator_llm
        )

    async def scorer(self, state: NewAgentStateSchema) -> ScorerOutputSchema:
        return await scorer(state, agent_config=self._config, llm=self._scorer_llm)

    async def uploader(self, state: NewAgentStateSchema) -> UploaderOutputSchema:
        return await uploader(state, background_uploader=self._background_uploader)

    async def drain_uploads(self) -> None:
        await self._background_uploader.drain()
        logger.success("[UPLOAD] Background uploader drained")

    async def _harvest_prefetch(self) -> dict[int, list[ComponentInfoSchema]]:
        if self._prefetch_task is None:
            return {}
        task, self._prefetch_task = self._prefetch_task, None
        try:
            return await task
        except Exception as exc:
            logger.warning(
                f"Context prefetch failed; falling back to live fetch: {exc!r}"
            )
            return {}

    def _start_prefetch(self) -> None:
        next_units = self._unit_iterator.peek_units(
            component_slug=self._config.component_slug,
            lang=self._config.target_lang,
            batch_size=self._config.batch_size,
            q="state:empty",
        )
        if not next_units:
            return
        self._prefetch_task = asyncio.create_task(
            context_collector(next_units, client=self._client)
        )

    async def aclose(self) -> None:
        await self.drain_uploads()
        await self._client.close()
