import asyncio
from typing import Any

from httpx import AsyncClient, Client
from loguru import logger

from src.agent.config import ConfigSchema
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
from src.agent.review import ReviewOutputSchema, ReviewPolicy
from src.models.agent import ComponentInfoSchema, NewAgentStateSchema
from src.services.weblate import AsyncWeblateClient


class WorkflowNodes:
    """Binds client/config/policy (and the once-built LLM runnables) to nodes.

    One instance serves an entire job. The three LLM runnables, the glossary
    snapshot and the Weblate connection pool are built once, not once per
    component — the automatic path translates many components through the
    same instance.
    """

    def __init__(
        self,
        client: AsyncWeblateClient,
        config: ConfigSchema,
        *,
        review: ReviewPolicy,
        owns_client: bool = True,
        http_client: Client | None = None,
        http_async_client: AsyncClient | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._review = review
        self._owns_client = owns_client
        self._unit_iterator = UnitIterator(client)
        self._translator_agent = build_translator_llm(
            config, http_client=http_client, http_async_client=http_async_client
        )
        self._tag_validator_llm = build_tag_validator_llm(
            config, http_client=http_client, http_async_client=http_async_client
        )
        self._scorer_llm = build_scorer_llm(
            config, http_client=http_client, http_async_client=http_async_client
        )
        self._glossaries: asyncio.Task[GlossaryLoaderOutputSchema] | None = None
        self._prefetch: dict[
            str, asyncio.Task[dict[int, list[ComponentInfoSchema]]]
        ] = {}
        self._background_uploader = BackgroundUploader(client)
        self.pattern_extractor = pattern_extractor

    async def glossary_loader(
        self, state: NewAgentStateSchema
    ) -> GlossaryLoaderOutputSchema:
        """Single-flight per-job load.

        There is no await between the None check and the task assignment,
        so concurrent components on one event loop share a single load
        instead of each issuing three full Weblate reads.
        """
        if self._glossaries is None:
            self._glossaries = asyncio.create_task(
                glossary_loader(client=self._client, agent_config=self._config)
            )
        return await self._glossaries

    async def fetch_empty(self, state: NewAgentStateSchema) -> FetchEmptyOutputSchema:
        return await fetch_empty(
            state, unit_iterator=self._unit_iterator, agent_config=self._config
        )

    async def context_collector(
        self, state: NewAgentStateSchema
    ) -> ContextResultsOutputSchema:
        prefetched = await self._harvest_prefetch(state.component_slug)
        hit = {u.id: prefetched[u.id] for u in state.to_translate if u.id in prefetched}
        missing = [u for u in state.to_translate if u.id not in prefetched]
        fresh = await context_collector(missing, client=self._client)
        self._start_prefetch(state.component_slug)
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

    async def review(self, state: NewAgentStateSchema) -> ReviewOutputSchema:
        return await self._review(state, agent_config=self._config)

    async def uploader(self, state: NewAgentStateSchema) -> UploaderOutputSchema:
        return await uploader(state, background_uploader=self._background_uploader)

    async def drain_uploads(self) -> None:
        await self._background_uploader.drain()
        logger.success("[UPLOAD] Background uploader drained")

    async def _harvest_prefetch(
        self, component_slug: str
    ) -> dict[int, list[ComponentInfoSchema]]:
        task = self._prefetch.pop(component_slug, None)
        if task is None:
            return {}
        try:
            return await task
        except Exception as exc:
            logger.warning(
                f"Context prefetch failed; falling back to live fetch: {exc!r}"
            )
            return {}

    def _start_prefetch(self, component_slug: str) -> None:
        next_units = self._unit_iterator.peek_units(
            component_slug=component_slug,
            lang=self._config.target_lang,
            batch_size=self._config.batch_size,
            q="state:empty",
        )
        if not next_units:
            return
        self._prefetch[component_slug] = asyncio.create_task(
            context_collector(next_units, client=self._client)
        )

    async def aclose(self) -> None:
        pending: list[asyncio.Task[Any]] = list(self._prefetch.values())
        if self._glossaries is not None:
            pending.append(self._glossaries)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._prefetch.clear()
        try:
            await self.drain_uploads()
        except BaseExceptionGroup:
            logger.exception("Background uploads failed during close")
        if self._owns_client:
            await self._client.close()
