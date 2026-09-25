import asyncio

import pytest

from src.agent.config import ConfigSchema
from src.agent.nodes.glossary_loader import load_glossaries
from src.models.glossary import GlossaryTerm
from src.models.weblate import CorpusUnitSchema, WeblateConfigSchema, WeblateUnitSchema
from src.services import glossary as module
from src.services.glossary import CustomGlossaryWriter, GlossarySnapshots
from src.services.weblate import AsyncWeblateClient

GLOSSARIES = {
    "base": {"Frag Grenade": "破片榴弹", "Acid Grenade": "酸液榴弹"},
    "mods": {"Tesla Grenade": "特斯拉榴弹"},
    "custom": {"Venom Grenade": "毒液榴弹"},
}


class FakeWeblate:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail: set[str] = set()
        self.gate: asyncio.Event | None = None

    async def list_units(
        self, slug: str, language: str, q: str = ""
    ) -> list[WeblateUnitSchema]:
        self.calls.append(slug)
        if self.gate is not None:
            await self.gate.wait()
        if slug in self.fail:
            raise RuntimeError(f"{slug} unavailable")
        return [
            WeblateUnitSchema(
                id=i, language_code=language, source=s, target=t, context=""
            )
            for i, (s, t) in enumerate(GLOSSARIES[slug].items())
        ]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncWeblateClient, FakeWeblate]:
    client = AsyncWeblateClient(
        WeblateConfigSchema(url="http://weblate", token="t", project_slug="p")
    )
    fake = FakeWeblate()
    monkeypatch.setattr(client, "list_units", fake.list_units)
    return client, fake


async def test_reuses_snapshots_within_ttl(fake, agent_config: ConfigSchema) -> None:
    client, weblate = fake
    cache = GlossarySnapshots(client, ttl_seconds=600)
    first = await load_glossaries(cache, agent_config)
    await load_glossaries(cache, agent_config)
    assert sorted(weblate.calls) == ["base", "custom", "mods"]
    assert first["patterns"]["{X} Grenade"][0].tgt_pattern == "{X}榴弹"


async def test_invalidate_reloads_only_that_glossary(
    fake, agent_config: ConfigSchema
) -> None:
    client, weblate = fake
    cache = GlossarySnapshots(client, ttl_seconds=600)
    await load_glossaries(cache, agent_config)
    cache.invalidate("custom")
    await load_glossaries(cache, agent_config)
    assert weblate.calls.count("custom") == 2
    assert weblate.calls.count("base") == 1


async def test_expired_snapshots_reload(
    fake, agent_config: ConfigSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, weblate = fake
    now = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    cache = GlossarySnapshots(client, ttl_seconds=600)
    await load_glossaries(cache, agent_config)
    now[0] += 601
    await load_glossaries(cache, agent_config)
    assert len(weblate.calls) == 6


async def test_failed_read_is_not_reused(fake, agent_config: ConfigSchema) -> None:
    client, weblate = fake
    cache = GlossarySnapshots(client, ttl_seconds=600)
    weblate.fail = {"mods"}
    with pytest.raises(RuntimeError):
        await load_glossaries(cache, agent_config)
    weblate.fail = set()
    await load_glossaries(cache, agent_config)
    assert weblate.calls.count("mods") == 2
    assert weblate.calls.count("base") == 1


async def test_concurrent_loads_share_reads_and_survive_cancel(
    fake, agent_config: ConfigSchema
) -> None:
    client, weblate = fake
    weblate.gate = asyncio.Event()
    cache = GlossarySnapshots(client, ttl_seconds=600)
    cancelled = asyncio.create_task(load_glossaries(cache, agent_config))
    survivor = asyncio.create_task(load_glossaries(cache, agent_config))
    while len(weblate.calls) < 3:  # both loads now wait on the shared reads
        await asyncio.sleep(0)
    cancelled.cancel()
    weblate.gate.set()
    result = await survivor
    assert len(weblate.calls) == 3
    assert "Frag Grenade" in result["base_glossary"]


async def test_writer_diffs_against_shared_snapshot_then_invalidates(
    fake, agent_config: ConfigSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, weblate = fake
    created: list[str] = []
    uploaded: list[CorpusUnitSchema] = []

    async def create_unit(slug: str, draft) -> None:
        created.append(draft.source)

    async def upload_targets(
        slug: str, language: str, units: list[CorpusUnitSchema]
    ) -> None:
        uploaded.extend(units)

    monkeypatch.setattr(client, "create_unit", create_unit)
    monkeypatch.setattr(client, "upload_targets", upload_targets)
    snapshots = GlossarySnapshots(client, ttl_seconds=600)
    await load_glossaries(snapshots, agent_config)
    writer = CustomGlossaryWriter(
        client, snapshots, component_slug="custom", target_lang="zh_Hans"
    )

    added, skipped = await writer.write(
        [
            GlossaryTerm(source="Venom Grenade", target="毒液榴弹", category="item"),
            GlossaryTerm(source="Smoke Grenade", target="烟雾榴弹", category="item"),
        ]
    )

    assert (added, skipped) == (1, 1)
    assert created == ["Smoke Grenade"]
    assert [u.target for u in uploaded] == ["烟雾榴弹"]
    assert weblate.calls.count("custom") == 1  # the diff reused the snapshot
    await load_glossaries(snapshots, agent_config)
    assert weblate.calls.count("custom") == 2  # the write invalidated it
    assert weblate.calls.count("base") == 1
