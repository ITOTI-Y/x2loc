from pathlib import Path, PurePosixPath

from src.core.workshop import component_slug
from src.jobs.pipeline import (
    COMPONENT_NAME_MAX_LENGTH,
    AssetWorkSchema,
    WorkshopPipeline,
    component_name,
)
from src.models.store import DEFAULT_PRIORITY
from src.models.weblate import CorpusUnitSchema
from src.models.workshop import LocalizationAssetSchema
from src.services.stores import merge


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

    found = WorkshopPipeline._with_shipped(w, {"held": {"upstream": "步枪"}})

    assert found == {
        "held": {"upstream": "步枪", "shipped": "模组步枪"},
        "lacking": {"shipped": "霰弹枪"},
    }
    assert merge(found, DEFAULT_PRIORITY) == {"held": "步枪", "lacking": "霰弹枪"}
