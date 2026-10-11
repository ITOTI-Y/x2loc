from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from src.core.workshop import component_slug
from src.jobs.pipeline import (
    COMPONENT_NAME_MAX_LENGTH,
    AssetWorkSchema,
    WorkshopPipeline,
    component_name,
)
from src.models.weblate import (
    CorpusUnitSchema,
    WeblateUnitPatchSchema,
    WeblateUnitSchema,
)
from src.models.workshop import LocalizationAssetSchema
from src.services.weblate import WEBLATE_STATE_EMPTY, WEBLATE_STATE_TRANSLATED


def asset(relative: str) -> LocalizationAssetSchema:
    path = PurePosixPath(relative)
    return LocalizationAssetSchema(
        source_path=Path(relative),
        relative_source_path=path,
        relative_target_path=path.with_suffix(".chn"),
        component_slug=component_slug("42", path),
    )


def test_component_name_keeps_short_names() -> None:
    assert component_name(asset("Localization/Foo.int"), "42-mod") == (
        "42-mod/Localization/Foo.int"
    )


def test_component_name_shortens_long_names_keeping_file_name() -> None:
    long_asset = asset("Localization/" + "Nested" * 20 + "/XComGame.int")
    name = component_name(long_asset, "42-long-war-of-the-chosen")
    assert len(name) == COMPONENT_NAME_MAX_LENGTH
    assert name.startswith(long_asset.component_slug)
    assert name.endswith("/XComGame.int")


class FakeWeblate:
    """Serves one component whose CSV export shows keys as sources."""

    config = SimpleNamespace(source_language="en")

    def __init__(self, sources: dict[str, str]) -> None:
        self.sources = sources
        self.patches: list[tuple[int, list[str], int]] = []

    async def download_units(self, slug: str, language: str) -> list[CorpusUnitSchema]:
        return [
            CorpusUnitSchema(context=c, source=c, target=s, note="")
            for c, s in self.sources.items()
        ]

    async def list_units(self, slug: str, language: str) -> list[WeblateUnitSchema]:
        base = 100 if language == "en" else 200
        return [
            WeblateUnitSchema(
                id=base + i, language_code=language, source=s, target="", context=c
            )
            for i, (c, s) in enumerate(self.sources.items())
        ]

    async def patch_unit(self, unit_id: int, patch: WeblateUnitPatchSchema) -> None:
        self.patches.append((unit_id, patch.target, patch.state))


def work(expected: dict[str, str]) -> AssetWorkSchema:
    return AssetWorkSchema.model_construct(
        asset=asset("Localization/Foo.int"), units=[], expected=expected
    )


def snapshot_of(targets: dict[str, str]) -> list[CorpusUnitSchema]:
    # Keys in the source column, as Weblate exports some units.
    return [
        CorpusUnitSchema(context=c, source=c, target=t, note="")
        for c, t in targets.items()
    ]


async def test_key_shaped_sources_are_read_from_the_source_file() -> None:
    weblate = FakeWeblate({"k1": "Tongue Grab"})
    pipeline = SimpleNamespace(_weblate=weblate)
    w = work({"k1": "Tongue Grab"})

    snapshot = await WorkshopPipeline._true_sources(
        pipeline,  # ty: ignore[invalid-argument-type]
        w,
        snapshot_of({"k1": "长舌拉拽"}),
    )
    previous = await WorkshopPipeline._clear_changed_sources(
        pipeline,  # ty: ignore[invalid-argument-type]
        w,
        "zh_Hans",
        snapshot,
    )

    assert [u.source for u in snapshot] == ["Tongue Grab"]
    assert previous == {}
    assert weblate.patches == []


async def test_changed_source_is_updated_and_its_translation_redone() -> None:
    weblate = FakeWeblate({"k1": "Old text"})
    pipeline = SimpleNamespace(_weblate=weblate)
    w = work({"k1": "New text"})

    snapshot = await WorkshopPipeline._true_sources(
        pipeline,  # ty: ignore[invalid-argument-type]
        w,
        snapshot_of({"k1": "旧译文"}),
    )
    previous = await WorkshopPipeline._clear_changed_sources(
        pipeline,  # ty: ignore[invalid-argument-type]
        w,
        "zh_Hans",
        snapshot,
    )

    assert previous == {"k1": "旧译文"}
    assert weblate.patches == [
        (100, ["New text"], WEBLATE_STATE_TRANSLATED),
        (200, [""], WEBLATE_STATE_EMPTY),
    ]


def test_overlay_fills_units_weblate_lacks_with_the_mods_translation() -> None:
    w = AssetWorkSchema.model_construct(
        asset=asset("Localization/Foo.int"),
        units=[
            CorpusUnitSchema(context=c, source=s, target=t, note="")
            for c, s, t in [
                ("held", "Rifle", "模组步枪"),
                ("lacking", "Shotgun", "霰弹枪"),
                ("broken", "Deals <Ability:X/> damage", "造成伤害"),
                ("untranslated", "Pistol", ""),
            ]
        ],
        expected={},
    )

    assert WorkshopPipeline._with_shipped(w, {"held": "步枪"}) == {
        "held": "步枪",
        "lacking": "霰弹枪",
    }
