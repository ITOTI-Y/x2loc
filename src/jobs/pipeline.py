from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
from loguru import logger
from openai import APIStatusError
from pydantic import ConfigDict

from src.agent.config import ConfigSchema
from src.agent.graph import build_graph, graph_recursion_limit
from src.agent.review import ThresholdReview, TranslationQualityError
from src.api.config import ServiceConfigSchema
from src.core.aligner import BilingualAligner
from src.core.artifact import ArtifactBuilder, ArtifactValidationError
from src.core.converter import CorpusConverter
from src.core.extractor import TermExtractor
from src.core.glossary import CustomGlossaryWriter
from src.core.parser import LocFileParser
from src.core.workshop import WorkshopInputError, discover_localization_assets
from src.jobs.manager import JobManager
from src.models._share import BaseSchema
from src.models.agent import NewAgentStateSchema
from src.models.file import LocalizationFile
from src.models.job import (
    ArtifactSchema,
    JobProgressSchema,
    JobStage,
    JobStatus,
    JobUpdateSchema,
    WorkshopJobRequestSchema,
)
from src.models.weblate import CorpusUnitSchema
from src.models.workshop import LocalizationAssetSchema, WorkshopItemSchema
from src.services.steam import SteamDownloader, SteamDownloadError
from src.services.weblate import AsyncWeblateClient, WeblateAPIError

# Ordered by specificity: WorkshopInputError is a ValueError subclass and must
# be matched before the ValueError catch-all.
ERROR_CODES: Final[tuple[tuple[type[Exception], str], ...]] = (
    (SteamDownloadError, "steam_download_failed"),
    (WorkshopInputError, "mod_content_invalid"),
    (TranslationQualityError, "translation_quality_failed"),
    (ArtifactValidationError, "artifact_failed"),
    (WeblateAPIError, "weblate_failed"),
    (APIStatusError, "llm_failed"),
    (ValueError, "invalid_request"),
)


def error_code(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return error_code(exc.exceptions[0])
    for kind, code in ERROR_CODES:
        if isinstance(exc, kind):
            return code
    return "internal_error"


class AssetWorkSchema(BaseSchema):
    """Everything derived from one source file, parsed exactly once."""

    model_config = ConfigDict(frozen=True)

    asset: LocalizationAssetSchema
    source_file: LocalizationFile
    units: list[CorpusUnitSchema]
    expected: dict[str, str]


class WorkshopPipeline:
    def __init__(
        self,
        *,
        config: ServiceConfigSchema,
        jobs: JobManager,
        steam: SteamDownloader,
        weblate: AsyncWeblateClient,
        glossary_writer: CustomGlossaryWriter,
        llm_clients: tuple[httpx.Client, httpx.AsyncClient],
    ) -> None:
        self._config = config
        self._jobs = jobs
        self._steam = steam
        self._weblate = weblate
        self._glossary_writer = glossary_writer
        self._llm_sync, self._llm_async = llm_clients
        self._parser = LocFileParser()
        self._aligner = BilingualAligner()
        self._converter = CorpusConverter()
        self._extractor = TermExtractor()
        self._artifact = ArtifactBuilder()

    async def run(self, job_id: str, request: WorkshopJobRequestSchema) -> None:
        """Run one job to a terminal state.

        Cancellation propagates untouched; `JobManager` owns the cancelled
        record so that only one writer decides the terminal status.
        """
        try:
            await self._run(job_id, request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = error_code(exc)
            logger.warning("Job {} failed: {} ({!r})", job_id, code, exc)
            self._jobs.update(
                job_id,
                JobUpdateSchema(status=JobStatus.FAILED, stage=None, error_code=code),
            )

    async def _run(self, job_id: str, request: WorkshopJobRequestSchema) -> None:
        self._jobs.update(
            job_id,
            JobUpdateSchema(status=JobStatus.RUNNING, stage=JobStage.DOWNLOADING),
        )
        item = await self._steam.download(request.workshop_id)

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.DISCOVERING))
        works = await asyncio.to_thread(self._prepare, item, request.target_lang)
        progress = JobProgressSchema(
            files_total=len(works),
            units_total=sum(len(work.units) for work in works),
        )
        self._jobs.update(job_id, JobUpdateSchema(progress=progress))

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.SYNCING_WEBLATE))
        async with asyncio.TaskGroup() as sync_group:
            for work in works:
                sync_group.create_task(
                    self._sync(work, request.target_lang, item.mod_info.namespace)
                )

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.TRANSLATING))
        translated = await self._translate(request, works)
        progress = progress.model_copy(
            update={"units_translated": translated, "files_completed": len(works)}
        )
        self._jobs.update(job_id, JobUpdateSchema(progress=progress))

        async with asyncio.TaskGroup() as readback_group:
            readback = [
                readback_group.create_task(
                    self._authoritative(work, request.target_lang)
                )
                for work in works
            ]
        authoritative = [task.result() for task in readback]

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.WRITING))
        overlay = self._config.work_root / job_id / "overlay"
        written = await asyncio.to_thread(
            self._write_all, works, authoritative, request.target_lang, overlay
        )

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.EXTRACTING_TERMS))
        added, skipped = await self._extract_terms(works, written, item, request)
        progress = progress.model_copy(
            update={"terms_added": added, "terms_skipped": skipped}
        )

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.PACKAGING))
        artifact_path = self._config.artifact_root / f"{job_id}.zip"
        size, digest = await asyncio.to_thread(
            self._artifact.package,
            outputs=[
                (work.asset.relative_target_path, path)
                for work, (path, _file) in zip(works, written, strict=True)
            ],
            artifact_path=artifact_path,
        )
        self._jobs.update(
            job_id,
            JobUpdateSchema(
                status=JobStatus.SUCCEEDED,
                stage=None,
                progress=progress,
                artifact=ArtifactSchema(path=artifact_path, sha256=digest, bytes=size),
            ),
        )

    def _prepare(
        self, item: WorkshopItemSchema, target_lang: str
    ) -> list[AssetWorkSchema]:
        """Parse and align every source file once, up front.

        Everything downstream reads from the returned objects; nothing
        re-reads `asset.source_path`.
        """
        works: list[AssetWorkSchema] = []
        for asset in discover_localization_assets(item):
            source_file = self._parser.parse(asset.source_path)
            target_file = (
                self._parser.parse(asset.existing_target_path)
                if asset.existing_target_path
                else None
            )
            corpus = self._aligner.align(
                source_file,
                target_file,
                target_lang=target_lang,
                mod_info=item.mod_info,
            )
            units = [
                CorpusUnitSchema.from_row(row)
                for row in self._converter.to_units(corpus)
            ]
            if not units:
                logger.info(
                    "Skipping {}: no translatable units",
                    asset.relative_source_path,
                )
                continue
            works.append(
                AssetWorkSchema(
                    asset=asset,
                    source_file=source_file,
                    units=units,
                    expected={unit.context: unit.source for unit in units},
                )
            )
        if not works:
            raise WorkshopInputError("mod contains no translatable localization units")
        return works

    async def _sync(
        self, work: AssetWorkSchema, target_lang: str, namespace: str
    ) -> None:
        # The slug is already unique per mod and file; the display name
        # carries the mod namespace because a bare relative path such as
        # "Localization/XComGame.int" is identical across most mods and
        # Weblate rejects duplicate component names within one project.
        await self._weblate.sync_corpus(
            work.asset.component_slug,
            name=f"{namespace}/{work.asset.relative_source_path.as_posix()}",
            language=target_lang,
            units=work.units,
            has_existing_target=work.asset.existing_target_path is not None,
        )

    async def _translate(
        self, request: WorkshopJobRequestSchema, works: list[AssetWorkSchema]
    ) -> int:
        """Translate every component through one graph and one node instance.

        Component concurrency is `llm_concurrency // batch_size` so that the
        in-flight LLM request ceiling stays at `llm_concurrency`: each
        component's own batch runs at most `batch_size` calls at a time.
        """
        agent_config = self._agent_config(request)
        graph, nodes = build_graph(
            agent_config,
            review=ThresholdReview(),
            client=self._weblate,
            http_client=self._llm_sync,
            http_async_client=self._llm_async,
        )
        limit = max(1, request.llm_concurrency // agent_config.batch_size)
        semaphore = asyncio.Semaphore(limit)

        async def translate_one(work: AssetWorkSchema) -> int:
            async with semaphore:
                final = await graph.ainvoke(
                    NewAgentStateSchema(component_slug=work.asset.component_slug),
                    config={
                        "configurable": {"thread_id": str(uuid4())},
                        "recursion_limit": graph_recursion_limit(
                            len(work.units),
                            batch_size=agent_config.batch_size,
                            max_attempts=agent_config.max_translation_attempts,
                        ),
                    },
                )
            stats = final["stats"]
            return stats["approved"] + stats["modified"] + stats["auto"]

        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(translate_one(work)) for work in works]
            await nodes.drain_uploads()
        finally:
            await nodes.aclose()
        return sum(task.result() for task in tasks)

    async def _authoritative(
        self, work: AssetWorkSchema, target_lang: str
    ) -> dict[str, str]:
        """Read back Weblate's own view of the component.

        Weblate is the authority: whatever it holds wins over the local
        `.chn`, so the overlay is built from this read, not from what the
        translator produced.
        """
        units = await self._weblate.list_units(work.asset.component_slug, target_lang)
        result: dict[str, str] = {}
        for unit in units:
            expected_source = work.expected.get(unit.context)
            if expected_source is None:
                continue
            if unit.context in result:
                raise ArtifactValidationError(
                    f"{work.asset.component_slug} returned a duplicate unit context"
                )
            if unit.source != expected_source:
                raise ArtifactValidationError(
                    f"{work.asset.component_slug} returned a stale source unit"
                )
            if unit.target.strip():
                result[unit.context] = unit.target
        missing = work.expected.keys() - result.keys()
        if missing:
            raise ArtifactValidationError(
                f"{work.asset.component_slug} has {len(missing)} untranslated units"
            )
        return result

    def _write_all(
        self,
        works: list[AssetWorkSchema],
        authoritative: list[dict[str, str]],
        target_lang: str,
        overlay: Path,
    ) -> list[tuple[Path, LocalizationFile]]:
        written: list[tuple[Path, LocalizationFile]] = []
        total_bytes = 0
        for work, translations in zip(works, authoritative, strict=True):
            path, target_file = self._artifact.write_target(
                asset=work.asset,
                source_file=work.source_file,
                translations=translations,
                target_lang=target_lang,
                staging_root=overlay,
            )
            total_bytes += path.stat().st_size
            if total_bytes > self._config.limits.max_total_bytes:
                raise ArtifactValidationError("generated overlay exceeds byte limits")
            written.append((path, target_file))
        return written

    async def _extract_terms(
        self,
        works: list[AssetWorkSchema],
        written: list[tuple[Path, LocalizationFile]],
        item: WorkshopItemSchema,
        request: WorkshopJobRequestSchema,
    ) -> tuple[int, int]:
        corpora = [
            self._aligner.align(
                work.source_file,
                target_file,
                target_lang=request.target_lang,
                mod_info=item.mod_info,
            )
            for work, (_path, target_file) in zip(works, written, strict=True)
        ]
        glossary = await asyncio.to_thread(self._extractor.extract, corpora)
        return await self._glossary_writer.write(glossary.terms)

    def _agent_config(self, request: WorkshopJobRequestSchema) -> ConfigSchema:
        """Merge the request over the service's `[agent]` defaults.

        Explicit request fields win; empty ones fall back to the TOML so
        the independently configured validation and scoring models apply
        to routine submissions.
        """
        defaults = self._config.agent
        api_key = (
            request.llm_api_key
            if request.llm_api_key.get_secret_value()
            else defaults.api_key
        )
        translation_model = request.translation_model or defaults.translation_model_name
        if not api_key.get_secret_value() or not translation_model:
            raise ValueError(
                "no LLM api key or translation model: provide them in the "
                "request or in the [agent] table of the service TOML"
            )
        base_url = (
            str(request.llm_api_base_url)
            if request.llm_api_base_url
            else defaults.base_url
        )
        return ConfigSchema(
            weblate=self._config.weblate,
            steam=self._config.steam,
            base_glossary_slug=self._config.glossary.base_slug,
            mods_glossary_slug=self._config.glossary.mods_slug,
            custom_glossary_slug=self._config.glossary.custom_slug,
            translation_model_name=translation_model,
            validate_model_name=request.validation_model
            or defaults.validate_model_name,
            scoring_model_name=request.scoring_model or defaults.scoring_model_name,
            base_url=base_url,
            api_key=api_key,
            batch_size=defaults.batch_size,
            auto_approve_threshold=defaults.auto_approve_threshold,
            target_lang=request.target_lang,
            max_concurrency=request.llm_concurrency,
        )
