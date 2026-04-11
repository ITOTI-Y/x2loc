from loguru import logger

from src.core._share import iter_compound_keys
from src.models.corpus import BilingualCorpus, BilingualEntry
from src.models.entry import EntrySchema
from src.models.file import LocalizationFile
from src.models.section import SectionHeader


class BilingualAligner:
    def align(
        self,
        source: LocalizationFile,
        target: LocalizationFile | None = None,
        target_lang: str | None = None,
    ):
        """Align two parsed localization files by compound key.

        Args:
            source: Source language file (parsed).
            target: Target language file (parsed). None for source-only corpus
                    (new localization with no existing target).
            target_lang: BCP-47 code for target language. Required when
                         target is None; otherwise inferred from target.lang.

        Returns:
            BilingualCorpus with aligned entries and diff lists.

        Raises:
            ValueError: target and target_lang are both None.
        """

        if target is None and target_lang is None:
            raise ValueError("Either target file or target_lang must be provided")

        source_index = self._build_index(source)

        if target is not None:
            target_index = self._build_index(target)
            effective_target_lang = target.lang
            effective_target_path = target.path
        else:
            target_index = {}
            effective_target_lang = target_lang
            effective_target_path = None

        entries: list[BilingualEntry] = []
        source_only_keys: list[str] = []
        matched_target_keys: set[str] = set()

        for compound_key, (src_entry, src_header) in source_index.items():
            if compound_key in target_index:
                tgt_entry, _ = target_index[compound_key]
                entries.append(
                    BilingualEntry(
                        compound_key=compound_key,
                        source=src_entry,
                        target=tgt_entry,
                        section_header=src_header,
                    )
                )
                matched_target_keys.add(compound_key)
            else:
                entries.append(
                    BilingualEntry(
                        compound_key=compound_key,
                        source=src_entry,
                        target=None,
                        section_header=src_header,
                    )
                )
                source_only_keys.append(compound_key)

        target_only_keys: list[str] = []
        for compound_key, (tgt_entry, tgt_header) in target_index.items():
            if compound_key not in matched_target_keys:
                target_only_keys.append(compound_key)

                entries.append(
                    BilingualEntry(
                        compound_key=compound_key,
                        source=tgt_entry,
                        target=tgt_entry,
                        section_header=tgt_header,
                    )
                )

        return BilingualCorpus(
            source_lang=source.lang,
            target_lang=effective_target_lang,  # type: ignore
            source_path=source.path,
            target_path=effective_target_path,
            entries=entries,
            source_only=source_only_keys,
            target_only=target_only_keys,
        )

    def _build_index(
        self, file: LocalizationFile
    ) -> dict[str, tuple[EntrySchema, SectionHeader]]:
        """Build compound_key -> (entry, section_header) mapping.

        Delegates the iteration contract (ordinal counters, key shape) to
        `iter_compound_keys`. Duplicate non-append keys within the same
        section: last wins + warn.
        """
        index: dict[str, tuple[EntrySchema, SectionHeader]] = {}

        for compound_key, entry, section in iter_compound_keys(file):
            if not entry.is_append and compound_key in index:
                logger.warning(f"Duplicate non-append key: {compound_key}, last wins")
            index[compound_key] = (entry, section.header)

        return index
