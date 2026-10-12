from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from src.core.workshop import component_slug
from src.models.weblate import (
    CorpusUnitSchema,
    WeblateTranslationStatsSchema,
    WeblateUnitPatchSchema,
    WeblateUnitSchema,
)
from src.models.workshop import AssetWorkSchema, LocalizationAssetSchema
from src.services.stores import LocalStore, UnitWrite
from src.services.weblate import WEBLATE_STATE_FUZZY, WEBLATE_STATE_TRANSLATED

LANG = "zh_Hans"
ASSET = LocalizationAssetSchema(
    source_path=Path("Localization/Foo.int"),
    relative_source_path=PurePosixPath("Localization/Foo.int"),
    relative_target_path=PurePosixPath("Localization/Foo.chn"),
    component_slug=component_slug("42", PurePosixPath("Localization/Foo.int")),
)
SLUG = ASSET.component_slug


class FakeWeblate:
    """One component, as (source, target) per unit; None when it is missing."""

    config = SimpleNamespace(source_language="en", project_slug="p")

    def __init__(self, units: dict[str, tuple[str, str]] | None) -> None:
        self.units = units
        self.patches: list[tuple[int, list[str], int]] = []
        self.synced: list[list[CorpusUnitSchema]] = []

    async def component_translations(
        self, slug: str
    ) -> dict[str, WeblateTranslationStatsSchema] | None:
        if self.units is None:
            return None
        total = len(self.units)
        return {
            LANG: WeblateTranslationStatsSchema(
                language_code=LANG, total=total, translated=total
            )
        }

    async def download_units(self, slug: str, language: str) -> list[CorpusUnitSchema]:
        return [
            CorpusUnitSchema(
                context=c, source=s, target=s if language == "en" else t, note=""
            )
            for c, (s, t) in (self.units or {}).items()
        ]

    async def list_units(
        self, slug: str, language: str, q: str = ""
    ) -> list[WeblateUnitSchema]:
        return [
            WeblateUnitSchema(
                id=200 + i, language_code=language, source=s, target=t, context=c
            )
            for i, (c, (s, t)) in enumerate((self.units or {}).items())
        ]

    async def search_units(self, q: str) -> list[WeblateUnitSchema]:
        return []

    async def sync_corpus(
        self, slug: str, *, name: str, language: str, units: list[CorpusUnitSchema]
    ) -> None:
        self.synced.append(units)
        if self.units is None:
            self.units = {u.context: (u.source, "") for u in units}

    async def patch_unit(self, unit_id: int, patch: WeblateUnitPatchSchema) -> None:
        self.patches.append((unit_id, patch.target, patch.state))

    async def add_unit_comment(self, unit_id: int, comment: str) -> None:
        pass


def work(sources: dict[str, str]) -> AssetWorkSchema:
    return AssetWorkSchema.model_construct(
        asset=ASSET,
        units=[
            CorpusUnitSchema(context=c, source=s, target="", note="")
            for c, s in sources.items()
        ],
        expected=sources,
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "store.sqlite"


def store_over(weblate: FakeWeblate, db: Path) -> LocalStore:
    return LocalStore(weblate, db)  # ty: ignore[invalid-argument-type]


async def test_prepare_skips_what_weblate_holds_usably(db: Path) -> None:
    weblate = FakeWeblate(
        {
            "held": ("Rifle", "步枪"),
            "stale": ("Old text", "旧译文"),
            "broken": ("Deals <Ability:X/> damage", "造成伤害"),
            "empty": ("Pistol", ""),
        }
    )
    store = store_over(weblate, db)
    w = work(
        {
            "held": "Rifle",
            "stale": "New text",
            "broken": "Deals <Ability:X/> damage",
            "empty": "Pistol",
        }
    )

    prepared = await store.prepare(w, name="n", language=LANG)
    pending = await store.pending(SLUG, LANG)

    assert prepared.needs_translation
    assert prepared.previous == {"stale": "旧译文"}
    assert sorted(u.context for u in pending) == ["broken", "empty", "stale"]
    assert await store.candidates(w, language=LANG) == {
        "held": {"upstream": "步枪"},
        "stale": {},
        "broken": {},
        "empty": {},
    }


async def test_a_component_weblate_lacks_is_all_pending(db: Path) -> None:
    store = store_over(FakeWeblate(None), db)

    prepared = await store.prepare(work({"a": "A"}), name="n", language=LANG)

    assert prepared.needs_translation
    assert [u.context for u in await store.pending(SLUG, LANG)] == ["a"]


async def test_written_and_edited_translations_become_candidates(db: Path) -> None:
    store = store_over(FakeWeblate(None), db)
    w = work({"a": "A", "b": "B", "c": "C"})
    await store.prepare(w, name="n", language=LANG)
    ids = {u.context: u.id for u in await store.pending(SLUG, LANG)}

    await store.write(UnitWrite(ids["a"], "甲", WEBLATE_STATE_TRANSLATED, ""))
    await store.write(UnitWrite(ids["b"], "乙", WEBLATE_STATE_FUZZY, "score 80"))
    store.edit(SLUG, LANG, "a", source="A", target="甲甲", upstream_at_read="")

    assert await store.candidates(w, language=LANG) == {
        "a": {"manual": "甲甲", "machine": "甲"},
        "b": {"needs_editing": "乙"},
        "c": {},
    }


async def test_local_translations_survive_a_reopen(db: Path) -> None:
    first = store_over(FakeWeblate(None), db)
    w = work({"a": "A"})
    await first.prepare(w, name="n", language=LANG)
    [unit] = await first.pending(SLUG, LANG)
    await first.write(UnitWrite(unit.id, "甲", WEBLATE_STATE_TRANSLATED, ""))
    await first.aclose()

    second = store_over(FakeWeblate(None), db)
    prepared = await second.prepare(w, name="n", language=LANG)

    assert not prepared.needs_translation
    assert await second.candidates(w, language=LANG) == {"a": {"machine": "甲"}}


async def test_upload_creates_the_component_and_holds_back_conflicts(
    db: Path,
) -> None:
    weblate = FakeWeblate(None)
    store = store_over(weblate, db)
    w = work({"a": "A", "b": "B"})
    await store.prepare(w, name="n", language=LANG)
    ids = {u.context: u.id for u in await store.pending(SLUG, LANG)}
    await store.write(UnitWrite(ids["a"], "甲", WEBLATE_STATE_TRANSLATED, ""))
    await store.write(UnitWrite(ids["b"], "乙", WEBLATE_STATE_TRANSLATED, ""))

    async def someone_translates_b(*args: object, **kwargs: object) -> None:
        await FakeWeblate.sync_corpus(weblate, *args, **kwargs)  # ty: ignore[invalid-argument-type]
        assert weblate.units is not None
        weblate.units["b"] = ("B", "别人的乙")

    weblate.sync_corpus = someone_translates_b  # ty: ignore[invalid-assignment]
    report = await store.upload(
        weblate,  # ty: ignore[invalid-argument-type]
        language=LANG,
    )

    assert (report.uploaded, report.conflicts) == (1, 1)
    assert [u.context for u in weblate.synced[0]] == ["a", "b"]
    assert weblate.patches == [(200, ["甲"], WEBLATE_STATE_TRANSLATED)]
    [conflict] = store.queue()
    assert (conflict.context, conflict.upload, conflict.upstream_now) == (
        "b",
        "conflict",
        "别人的乙",
    )


async def test_resolving_a_conflict_requeues_or_drops_the_local_one(
    db: Path,
) -> None:
    weblate = FakeWeblate({"a": ("A", ""), "b": ("B", "")})
    store = store_over(weblate, db)
    w = work({"a": "A", "b": "B"})
    await store.prepare(w, name="n", language=LANG)
    ids = {u.context: u.id for u in await store.pending(SLUG, LANG)}
    await store.write(UnitWrite(ids["a"], "甲", WEBLATE_STATE_TRANSLATED, ""))
    await store.write(UnitWrite(ids["b"], "乙", WEBLATE_STATE_TRANSLATED, ""))
    weblate.units = {"a": ("A", "别人的甲"), "b": ("B", "别人的乙")}
    await store.upload(weblate, language=LANG)  # ty: ignore[invalid-argument-type]

    store.resolve(SLUG, LANG, "a", "machine", keep="local")
    store.resolve(SLUG, LANG, "b", "machine", keep="upstream")
    report = await store.upload(weblate, language=LANG)  # ty: ignore[invalid-argument-type]

    assert (report.uploaded, report.conflicts) == (1, 0)
    assert weblate.patches[-1] == (200, ["甲"], WEBLATE_STATE_TRANSLATED)
    assert store.queue() == []


async def test_a_hand_edit_replaces_the_queued_machine_translation(db: Path) -> None:
    store = store_over(FakeWeblate(None), db)
    await store.prepare(work({"a": "A"}), name="n", language=LANG)
    [unit] = await store.pending(SLUG, LANG)
    await store.write(UnitWrite(unit.id, "甲", WEBLATE_STATE_FUZZY, "score 70"))

    store.edit(SLUG, LANG, "a", source="A", target="甲甲", upstream_at_read="")

    assert [(r.origin, r.target) for r in store.queue()] == [("manual", "甲甲")]
    assert store.review(LANG) == []
