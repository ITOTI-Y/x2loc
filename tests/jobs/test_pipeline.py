from pathlib import Path, PurePosixPath

from src.core.workshop import component_slug
from src.jobs.pipeline import COMPONENT_NAME_MAX_LENGTH, component_name
from src.models.workshop import LocalizationAssetSchema


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
