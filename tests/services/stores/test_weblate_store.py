from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from src.core.workshop import component_slug
from src.models.weblate import (
    CorpusUnitSchema,
    WeblateUnitPatchSchema,
    WeblateUnitSchema,
)
from src.models.workshop import AssetWorkSchema, LocalizationAssetSchema
from src.services.stores import Candidates, UnitWrite, WeblateStore, merge
from src.services.stores.upstream import true_sources
from src.services.weblate import (
    WEBLATE_STATE_EMPTY,
    WEBLATE_STATE_FUZZY,
    WEBLATE_STATE_TRANSLATED,
)

LANG = "zh_Hans"
ASSET = LocalizationAssetSchema(
    source_path=Path("Localization/Foo.int"),
    relative_source_path=PurePosixPath("Localization/Foo.int"),
    relative_target_path=PurePosixPath("Localization/Foo.chn"),
    component_slug=component_slug("42", PurePosixPath("Localization/Foo.int")),
)
SLUG = ASSET.component_slug


class FakeWeblate:
    """One component; its CSV export shows each unit's key as its source."""

    config = SimpleNamespace(source_language="en", project_slug="p")

    def __init__(self, sources: dict[str, str], targets: dict[str, str]) -> None:
        self.sources = sources
        self.targets = targets
        self.fuzzy: set[str] = set()
        self.patches: list[tuple[int, list[str], int]] = []
        self.downloads = 0

    async def download_units(self, slug: str, language: str) -> list[CorpusUnitSchema]:
        self.downloads += 1
        values = self.sources if language == "en" else self.targets
        return [
            CorpusUnitSchema(context=c, source=c, target=v, note="")
            for c, v in values.items()
        ]

    async def list_units(
        self, slug: str, language: str, q: str = ""
    ) -> list[WeblateUnitSchema]:
        base = 100 if language == "en" else 200
        return [
            WeblateUnitSchema(
                id=base + i, language_code=language, source=s, target="", context=c
            )
            for i, (c, s) in enumerate(self.sources.items())
        ]

    async def search_units(self, q: str) -> list[WeblateUnitSchema]:
        return [
            WeblateUnitSchema(
                id=300,
                language_code=LANG,
                source="",
                target="",
                context=context,
                translation=f"https://w/api/translations/p/{SLUG}/{LANG}/",
            )
            for context in self.fuzzy
        ]

    async def patch_unit(self, unit_id: int, patch: WeblateUnitPatchSchema) -> None:
        self.patches.append((unit_id, patch.target, patch.state))

    async def add_unit_comment(self, unit_id: int, comment: str) -> None:
        pass

    async def sync_corpus(self, slug: str, **_: object) -> None:
        return None


def work(expected: dict[str, str]) -> AssetWorkSchema:
    return AssetWorkSchema.model_construct(asset=ASSET, units=[], expected=expected)


async def test_key_shaped_sources_are_read_from_the_source_file() -> None:
    weblate = FakeWeblate({"k1": "Tongue Grab"}, {"k1": "长舌拉拽"})
    w = work({"k1": "Tongue Grab"})

    snapshot = await true_sources(
        weblate,  # ty: ignore[invalid-argument-type]
        w,
        await weblate.download_units(SLUG, LANG),
    )
    prepared = await WeblateStore(weblate).prepare(  # ty: ignore[invalid-argument-type]
        w, name="n", language=LANG
    )

    assert [u.source for u in snapshot] == ["Tongue Grab"]
    assert prepared.previous == {}
    assert not prepared.needs_translation
    assert weblate.patches == []


async def test_changed_source_is_updated_and_its_translation_redone() -> None:
    weblate = FakeWeblate({"k1": "Old text"}, {"k1": "旧译文"})

    prepared = await WeblateStore(weblate).prepare(  # ty: ignore[invalid-argument-type]
        work({"k1": "New text"}), name="n", language=LANG
    )

    assert prepared.previous == {"k1": "旧译文"}
    assert prepared.needs_translation
    assert weblate.patches == [
        (100, ["New text"], WEBLATE_STATE_TRANSLATED),
        (200, [""], WEBLATE_STATE_EMPTY),
    ]


async def test_candidates_reuse_an_untouched_snapshot_and_mark_needs_editing() -> None:
    weblate = FakeWeblate({"a": "A", "b": "B"}, {"a": "甲", "b": "乙"})
    weblate.fuzzy = {"b"}
    store = WeblateStore(weblate)  # ty: ignore[invalid-argument-type]
    w = work({"a": "A", "b": "B"})

    await store.prepare(w, name="n", language=LANG)
    downloads = weblate.downloads
    found = await store.candidates(w, language=LANG)

    assert found == {"a": {"upstream": "甲"}, "b": {"needs_editing": "乙"}}
    assert weblate.downloads == downloads


async def test_a_write_in_this_run_decides_needs_editing() -> None:
    weblate = FakeWeblate({"a": "A"}, {"a": ""})
    store = WeblateStore(weblate)  # ty: ignore[invalid-argument-type]
    w = work({"a": "A"})

    [unit] = await store.pending(SLUG, LANG)
    await store.write(UnitWrite(unit.id, "甲", WEBLATE_STATE_FUZZY, "why"))
    weblate.targets["a"] = "甲"

    assert await store.candidates(w, language=LANG) == {"a": {"needs_editing": "甲"}}


def test_merge_takes_the_first_kind_in_priority() -> None:
    found: Candidates = {
        "a": {"upstream": "上游", "machine": "机器"},
        "b": {"shipped": "自带"},
        "c": {},
    }

    assert merge(found, ["machine", "upstream", "shipped"]) == {
        "a": "机器",
        "b": "自带",
    }
