from __future__ import annotations

import asyncio
import math
import shutil
import sys
from pathlib import Path
from typing import Final, Protocol, runtime_checkable
from uuid import uuid4

import httpx
from httpx2 import TransportError
from loguru import logger
from pydantic import ConfigDict

from src.agent.config import ConfigSchema, build_agent_config
from src.config import ServiceConfigSchema
from src.core.aligner import BilingualAligner
from src.core.artifact import ArtifactBuilder, ArtifactValidationError
from src.core.converter import CorpusConverter
from src.core.extractor import TermExtractor
from src.core.parser import LocFileParser
from src.core.placeholders import validate_tags
from src.core.workshop import WorkshopInputError, discover_localization_assets
from src.jobs.manager import JobManager
from src.models._share import BaseSchema
from src.models.agent import NewAgentStateSchema
from src.models.file import LocalizationFile
from src.models.glossary import Glossary
from src.models.job import (
    ArtifactSchema,
    JobProgressSchema,
    JobStage,
    JobStatus,
    JobUpdateSchema,
    WorkshopJobRequestSchema,
)
from src.models.weblate import CorpusUnitSchema, WeblateUnitPatchSchema
from src.models.workshop import LocalizationAssetSchema, WorkshopItemSchema
from src.services.glossary import GlossarySource, GlossaryWriter
from src.services.steam import SteamDownloadError
from src.services.weblate import (
    WEBLATE_STATE_EMPTY,
    AsyncWeblateClient,
    WeblateAPIError,
)

# Ordered by specificity: WorkshopInputError is a ValueError subclass and must
# be matched before the ValueError catch-all.
ERROR_CODES: Final[tuple[tuple[type[Exception], str], ...]] = (
    (SteamDownloadError, "steam_download_failed"),
    (WorkshopInputError, "mod_content_invalid"),
    (ArtifactValidationError, "artifact_failed"),
    (WeblateAPIError, "weblate_failed"),
    (TransportError, "weblate_failed"),
    (ValueError, "invalid_request"),
)


COMPONENT_NAME_MAX_LENGTH: Final[int] = 100


def error_code(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return error_code(exc.exceptions[0])
    # The OpenAI SDK is loaded only with the translation graph; until then
    # no LLM error can exist, so the check needs no import of its own.
    openai = sys.modules.get("openai")
    if openai is not None and isinstance(exc, openai.APIStatusError):
        return "llm_failed"
    for kind, code in ERROR_CODES:
        if isinstance(exc, kind):
            return code
    return "internal_error"


class WorkshopSource(Protocol):
    """Where a job obtains the Workshop item it translates."""

    async def download(self, workshop_id: str) -> WorkshopItemSchema: ...


@runtime_checkable
class PreparedWorkshopSource(WorkshopSource, Protocol):
    """A source that already parsed its items, so jobs need not again."""

    def prepared_works(
        self, workshop_id: str, target_lang: str
    ) -> list[AssetWorkSchema] | None: ...


class AssetWorkSchema(BaseSchema):
    """Everything derived from one source file, parsed exactly once."""

    model_config = ConfigDict(frozen=True)

    asset: LocalizationAssetSchema
    source_file: LocalizationFile
    units: list[CorpusUnitSchema]
    expected: dict[str, str]


def component_name(asset: LocalizationAssetSchema, namespace: str) -> str:
    """Weblate display name: `namespace/path`, or `slug:...path-tail` if too long.

    The fallback stays unique because the slug is, and keeps the end of the
    path, where the file name is.
    """
    path = asset.relative_source_path.as_posix()
    name = f"{namespace}/{path}"
    if len(name) <= COMPONENT_NAME_MAX_LENGTH:
        return name
    prefix = f"{asset.component_slug}:..."
    return prefix + path[-(COMPONENT_NAME_MAX_LENGTH - len(prefix)) :]


def prepare_works(item: WorkshopItemSchema, target_lang: str) -> list[AssetWorkSchema]:
    """Parse and align every source file once, up front.

    Everything downstream reads from the returned objects; nothing
    re-reads `asset.source_path`.

    Raises:
        WorkshopInputError: If no file yields a translatable unit.
    """
    parser = LocFileParser()
    aligner = BilingualAligner()
    converter = CorpusConverter()
    works: list[AssetWorkSchema] = []
    for asset in discover_localization_assets(item):
        source_file = parser.parse(asset.source_path)
        target_file = (
            parser.parse(asset.existing_target_path)
            if asset.existing_target_path
            else None
        )
        corpus = aligner.align(
            source_file,
            target_file,
            target_lang=target_lang,
            mod_info=item.mod_info,
        )
        units = [CorpusUnitSchema.from_row(row) for row in converter.to_units(corpus)]
        if not units:
            logger.info(
                "Skipping {}: no translatable units", asset.relative_source_path
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


class WorkshopPipeline:
    def __init__(
        self,
        *,
        config: ServiceConfigSchema,
        jobs: JobManager,
        source: WorkshopSource,
        weblate: AsyncWeblateClient,
        glossary_writer: GlossaryWriter,
        glossaries: GlossarySource,
        llm_client: httpx.AsyncClient,
    ) -> None:
        self._config = config
        self._jobs = jobs
        self._source = source
        self._weblate = weblate
        self._glossary_writer = glossary_writer
        self._glossaries = glossaries
        self._llm_client = llm_client
        self._aligner = BilingualAligner()
        self._extractor = TermExtractor()
        self._artifact = ArtifactBuilder()
        # Shared by every job of this pipeline, so concurrent jobs keep the
        # in-flight LLM ceiling of one; keyed by the per-job limit.
        self._translate_slots: dict[int, asyncio.Semaphore] = {}

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
        item = await self._source.download(request.workshop_id)

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.DISCOVERING))
        works = (
            self._source.prepared_works(request.workshop_id, request.target_lang)
            if isinstance(self._source, PreparedWorkshopSource)
            else None
        )
        if works is None:
            works = await asyncio.to_thread(prepare_works, item, request.target_lang)
        progress = JobProgressSchema(
            files_total=len(works),
            units_total=sum(len(work.units) for work in works),
        )
        self._jobs.update(job_id, JobUpdateSchema(progress=progress))

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.SYNCING_WEBLATE))
        async with asyncio.TaskGroup() as sync_group:
            syncs = [
                sync_group.create_task(
                    self._sync(work, request.target_lang, item.mod_info.namespace)
                )
                for work in works
            ]
        snapshots = [task.result() for task in syncs]
        # A component whose post-sync snapshot holds no empty target has
        # nothing for the graph to fetch; skipping it saves a round trip and
        # keeps its snapshot valid for the readback below.
        pending = [
            work
            for work, snapshot in zip(works, snapshots, strict=True)
            if snapshot is None or any(not unit.target for unit in snapshot)
        ]

        self._jobs.update(job_id, JobUpdateSchema(stage=JobStage.TRANSLATING))
        translated = await self._translate(request, pending)
        progress = progress.model_copy(
            update={"units_translated": translated, "files_completed": len(works)}
        )
        self._jobs.update(job_id, JobUpdateSchema(progress=progress))

        pending_slugs = {work.asset.component_slug for work in pending}

        async def read_back(
            work: AssetWorkSchema, snapshot: list[CorpusUnitSchema] | None
        ) -> dict[str, str]:
            slug = work.asset.component_slug
            if snapshot is None or slug in pending_slugs:
                snapshot = await self._weblate.download_units(slug, request.target_lang)
            return self._authoritative(work, snapshot)

        async with asyncio.TaskGroup() as readback_group:
            readback = [
                readback_group.create_task(read_back(work, snapshot))
                for work, snapshot in zip(works, snapshots, strict=True)
            ]
        authoritative = [task.result() for task in readback]
        progress = progress.model_copy(
            update={
                "units_untranslated": sum(
                    len(work.expected) - len(translations)
                    for work, translations in zip(works, authoritative, strict=True)
                )
            }
        )

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

    async def _sync(
        self, work: AssetWorkSchema, target_lang: str, namespace: str
    ) -> list[CorpusUnitSchema] | None:
        """Sync one component; return its target units if still current.

        The snapshot is None once mismatched tags were cleared from it.
        """
        # The slug is already unique per mod and file; the display name
        # carries the mod namespace because a bare relative path such as
        # "Localization/XComGame.int" is identical across most mods and
        # Weblate rejects duplicate component names within one project.
        snapshot = await self._weblate.sync_corpus(
            work.asset.component_slug,
            name=component_name(work.asset, namespace),
            language=target_lang,
            units=work.units,
            has_existing_target=work.asset.existing_target_path is not None,
        )
        if snapshot is None:
            snapshot = await self._weblate.download_units(
                work.asset.component_slug, target_lang
            )
        if await self._clear_tag_mismatches(work, target_lang, snapshot):
            return None
        return snapshot

    async def _clear_tag_mismatches(
        self,
        work: AssetWorkSchema,
        target_lang: str,
        snapshot: list[CorpusUnitSchema],
    ) -> bool:
        """Empty held translations whose tags disagree with their source.

        Such a translation, typically shipped in the mod's own `.chn`, would
        fail the overlay check; emptied, it is retranslated like any
        untranslated unit. Returns whether anything was cleared.
        """
        slug = work.asset.component_slug
        broken_contexts = {
            unit.context
            for unit in snapshot
            if unit.context in work.expected
            and unit.target.strip()
            and not validate_tags(unit.source, unit.target)[0]
        }
        if not broken_contexts:
            return False
        # Patching needs unit ids, which only the (slow) units API carries.
        broken = [
            unit
            for unit in await self._weblate.list_units(slug, target_lang)
            if unit.context in broken_contexts
        ]
        for unit in broken:
            await self._weblate.patch_unit(
                unit.id, WeblateUnitPatchSchema(target=[""], state=WEBLATE_STATE_EMPTY)
            )
        logger.warning(
            "Cleared {} translations with mismatched tags in {}", len(broken), slug
        )
        return True

    async def _translate(
        self, request: WorkshopJobRequestSchema, works: list[AssetWorkSchema]
    ) -> int:
        """Translate every component through one graph and one node instance.

        A component's batch sends `batch_size / units_per_request` requests at
        once, so component concurrency is their quotient into
        `llm_concurrency`, keeping the in-flight request ceiling there.
        """
        if not works:
            return 0
        # Deferred: the LangChain/LangGraph stack costs ~1.5 s to import, and
        # a run over already translated mods never needs it.
        from src.agent.graph import build_graph, graph_recursion_limit
        from src.agent.review import ThresholdReview

        agent_config = self._agent_config(request)
        graph, nodes = build_graph(
            agent_config,
            review=ThresholdReview(),
            client=self._weblate,
            glossaries=self._glossaries,
            http_async_client=self._llm_client,
        )
        requests_per_batch = math.ceil(
            agent_config.batch_size / agent_config.units_per_request
        )
        limit = max(1, request.llm_concurrency // requests_per_batch)
        semaphore = self._translate_slots.setdefault(limit, asyncio.Semaphore(limit))

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

    @staticmethod
    def _authoritative(
        work: AssetWorkSchema, units: list[CorpusUnitSchema]
    ) -> dict[str, str]:
        """Weblate's own view of the component, from a fresh read.

        Weblate is the authority: whatever it holds wins over the local
        `.chn`, so the overlay is built from this read, not from what the
        translator produced. Units still empty here (skipped by the quality
        gate) are absent from the result, and the overlay keeps their source.
        """
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
        missing = len(work.expected) - len(result)
        if missing:
            logger.warning(
                "{} has {} untranslated units; the overlay keeps their source",
                work.asset.component_slug,
                missing,
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
        def align_and_extract() -> Glossary:
            corpora = [
                self._aligner.align(
                    work.source_file,
                    target_file,
                    target_lang=request.target_lang,
                    mod_info=item.mod_info,
                )
                for work, (_path, target_file) in zip(works, written, strict=True)
            ]
            return self._extractor.extract(corpora)

        glossary = await asyncio.to_thread(align_and_extract)
        return await self._glossary_writer.write(glossary.terms)

    def _agent_config(self, request: WorkshopJobRequestSchema) -> ConfigSchema:
        """Merge the request over the service's `[agent]` defaults.

        Explicit request fields win; empty ones fall back to the TOML so
        the independently configured validation and scoring models apply
        to routine submissions.
        """
        defaults = self._config.agent
        agent = defaults.model_copy(
            update={
                "api_key": request.llm_api_key
                if request.llm_api_key.get_secret_value()
                else defaults.api_key,
                "translation_model_name": request.translation_model
                or defaults.translation_model_name,
                "validate_model_name": request.validation_model
                or defaults.validate_model_name,
                "scoring_model_name": request.scoring_model
                or defaults.scoring_model_name,
                "base_url": str(request.llm_api_base_url)
                if request.llm_api_base_url
                else defaults.base_url,
            }
        )
        return build_agent_config(
            self._config,
            agent,
            target_lang=request.target_lang,
            max_concurrency=request.llm_concurrency,
        )


def reset_work_dirs(config: ServiceConfigSchema) -> None:
    """Wipe the work and artifact roots at startup.

    Failing to wipe must stop startup: booting on a half-cleared directory
    would serve stale artifacts from a forgotten process.
    """
    for path in (config.work_root, config.artifact_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
