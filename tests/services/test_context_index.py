import asyncio
import math

import pytest

from src.models.weblate import CorpusUnitSchema
from src.services.context_index import ContextIndex, ContextIndexSource
from src.services.weblate import WeblateAPIError


def row(context: str, source: str, target: str = "") -> CorpusUnitSchema:
    return CorpusUnitSchema(context=context, source=source, target=target, note="")


TEMPLATE = [
    row("Other X2Item::FriendlyName", "Unrelated"),
    row("Grenade X2Item::FriendlyName", "Plasma Grenade", "等离子手雷"),
    row("Grenade X2Item::BriefSummary", "Explodes.", "爆炸。"),
    row("Grenade X2Item::TacticalText", "<b>Boom</b>", "<b>轰</b>"),
]


def test_lookup_returns_other_components_with_same_template_neighbours() -> None:
    index = ContextIndex()
    index.put("mod-a", TEMPLATE)
    index.put("mod-self", [row("Grenade X2Item::FriendlyName", "Plasma Grenade")])

    [hit] = index.lookup("Plasma Grenade", exclude_slug="mod-self")

    assert (hit.slug, hit.key, hit.position) == ("mod-a", "Grenade X2Item", 2)
    assert hit.unit.target == "等离子手雷"
    assert [n.source for n in hit.nearby] == [
        "Plasma Grenade",
        "Explodes.",
        "<b>Boom</b>",
    ]


def test_lookup_matches_markup_verbatim() -> None:
    index = ContextIndex()
    index.put("mod-a", TEMPLATE)

    assert [h.slug for h in index.lookup("<b>Boom</b>", exclude_slug="")] == ["mod-a"]
    assert index.lookup("Boom", exclude_slug="") == []


def test_lookup_caps_matches_per_component_and_in_total() -> None:
    index = ContextIndex()
    for n in range(4):
        index.put(f"mod-{n}", [row(f"K{i}::F", "Same") for i in range(5)])

    hits = index.lookup("Same", exclude_slug="")

    assert [h.slug for h in hits] == ["mod-0"] * 3 + ["mod-1"] * 3


def test_put_replaces_a_component() -> None:
    index = ContextIndex()
    index.put("mod-a", [row("K::F", "Old")])
    index.put("mod-a", [row("K::F", "New")])

    assert index.lookup("Old", exclude_slug="") == []
    assert [h.slug for h in index.lookup("New", exclude_slug="")] == ["mod-a"]


class FakeWeblate:
    def __init__(self, files: dict[str, list[CorpusUnitSchema] | Exception]) -> None:
        self.files = files
        self.downloaded: list[str] = []
        self.listings = 0
        self.release = asyncio.Event()
        self.release.set()

    async def list_component_slugs(self) -> list[str]:
        self.listings += 1
        return ["glossary-custom", *self.files]

    async def download_units(self, slug: str, language: str) -> list[CorpusUnitSchema]:
        self.downloaded.append(slug)
        await self.release.wait()
        result = self.files[slug]
        if isinstance(result, Exception):
            raise result
        return result


def source_over(client: FakeWeblate) -> ContextIndexSource:
    return ContextIndexSource(
        client,  # ty: ignore[invalid-argument-type]
        language="zh_Hans",
        ttl_seconds=math.inf,
    )


async def test_index_downloads_unsynced_components_once_and_skips_glossaries() -> None:
    client = FakeWeblate(
        {
            "mod-a": TEMPLATE,
            "mod-synced": [row("K::F", "From sync")],
            "mod-no-zh": WeblateAPIError(404, "no translation"),
        }
    )
    context = source_over(client)
    context.update("mod-synced", [row("K::F", "From sync")])

    index = await context.index()
    again = await context.index()

    assert index is again
    assert client.listings == 1
    assert sorted(client.downloaded) == ["mod-a", "mod-no-zh"]
    assert "mod-no-zh" not in index
    assert [h.slug for h in index.lookup("Plasma Grenade", exclude_slug="")] == [
        "mod-a"
    ]


async def test_component_synced_during_the_build_wins_over_its_download() -> None:
    client = FakeWeblate({"mod-a": [row("K::F", "Stale")]})
    client.release.clear()
    context = source_over(client)

    build = asyncio.create_task(context.index())
    await asyncio.sleep(0)
    context.update("mod-a", [row("K::F", "Fresh")])
    client.release.set()
    index = await build

    assert index.lookup("Stale", exclude_slug="") == []
    assert [h.slug for h in index.lookup("Fresh", exclude_slug="")] == ["mod-a"]


async def test_failed_build_propagates_and_is_retried() -> None:
    client = FakeWeblate({"mod-a": WeblateAPIError(502, "down")})
    context = source_over(client)

    with pytest.raises(WeblateAPIError):
        await context.index()
    client.files["mod-a"] = TEMPLATE
    index = await context.index()

    assert client.listings == 2
    assert "mod-a" in index
