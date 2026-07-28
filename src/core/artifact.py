import hashlib
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from src.agent.tools import validate_tags
from src.core.converter import CorpusConverter
from src.core.loc_writer import LocFileWriter
from src.core.parser import LocFileParser
from src.models.file import LocalizationFile
from src.models.workshop import TARGET_SUFFIX, LocalizationAssetSchema

ZIP_COMPRESS_LEVEL = 6
HASH_CHUNK_BYTES = 1 << 20


class ArtifactValidationError(RuntimeError):
    """Generated overlay does not faithfully mirror its source file."""


def _entry_rows(file: LocalizationFile) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for section in file.sections:
        for entry in section.entries:
            rows.append((section.header.raw, entry.key, entry.value))
            for field in entry.struct_fields or []:
                rows.append(
                    (section.header.raw, f"{entry.key}.{field.key}", field.value)
                )
    return rows


class ArtifactBuilder:
    def __init__(self) -> None:
        self._parser = LocFileParser()
        self._writer = LocFileWriter()
        self._converter = CorpusConverter()

    def write_target(
        self,
        *,
        asset: LocalizationAssetSchema,
        source_file: LocalizationFile,
        translations: dict[str, str],
        target_lang: str,
        staging_root: Path,
    ):
        target_path = staging_root.joinpath(*asset.relative_target_path.parts)
        target_file = self._converter.build_target_file(
            source=source_file,
            translations=translations,
            target_lang=target_lang,
            target_path=target_path,
        )
        self._writer.write(file=target_file, path=target_path)
        self._verify(source_file, target_path, target_lang)
        return target_path, target_file

    def _verify(
        self, source_file: LocalizationFile, target_path: Path, target_lang: str
    ):
        reparsed = self._parser.parse(target_path, lang_override=target_lang)
        source_rows = _entry_rows(source_file)
        target_rows = _entry_rows(reparsed)
        if len(source_rows) != len(target_rows):
            raise ArtifactValidationError(
                f"{target_path.name} has {len(target_rows)} entries, "
                f"source has {len(source_rows)}"
            )
        for (section, key, source_value), (
            target_section,
            target_key,
            target_value,
        ) in zip(source_rows, target_rows, strict=True):
            if (section, key) != (target_section, target_key):
                raise ArtifactValidationError(
                    f"{target_path.name} structure diverged at {section}/{key}"
                )
            valid, missing, extra = validate_tags(source_value, target_value)
            if not valid:
                raise ArtifactValidationError(
                    f"{target_path.name} placeholder mismatch at {section}/{key}: "
                    f"missing={sorted(missing)} extra={sorted(extra)}"
                )

    def package(
        self,
        *,
        outputs: list[tuple[PurePosixPath, Path]],
        artifact_path: Path,
    ) -> tuple[int, str]:
        if not outputs:
            raise ArtifactValidationError("cannot package an empty overlay")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(".zip.part")
        seen: set[str] = set()
        try:
            with zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=ZIP_COMPRESS_LEVEL,
            ) as archive:
                for relative_target, staged in sorted(
                    outputs, key=lambda pair: pair[0].as_posix()
                ):
                    name = relative_target.as_posix()
                    windows_path = PureWindowsPath(name)
                    if (
                        relative_target.is_absolute()
                        or ".." in relative_target.parts
                        or windows_path.is_absolute()
                        or windows_path.drive
                    ):
                        raise ArtifactValidationError(f"unsafe ZIP entry: {name}")
                    if relative_target.suffix.casefold() != TARGET_SUFFIX:
                        raise ArtifactValidationError(f"unexpected ZIP entry: {name}")

                    collision_key = windows_path.as_posix().casefold()
                    if collision_key in seen:
                        raise ArtifactValidationError(f"duplicate ZIP entry: {name}")
                    if staged.is_symlink() or not staged.is_file():
                        raise ArtifactValidationError(f"unsafe ZIP input: {name}")
                    seen.add(collision_key)
                    archive.write(staged, name)
            temporary.replace(artifact_path)
        finally:
            temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        size = 0
        with artifact_path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
        return size, digest.hexdigest()
