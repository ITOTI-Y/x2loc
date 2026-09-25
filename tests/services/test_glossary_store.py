import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.models.glossary import GlossaryTerm
from src.models.weblate import CorpusUnitSchema, WeblateConfigSchema, WeblateUnitSchema
from src.services.glossary_store import (
    DeferredGlossaryWriter,
    LocalGlossaryStore,
    StoredGlossarySchema,
)
from src.services.weblate import AsyncWeblateClient, WeblateAPIError


def unit(unit_id: int, source: str, target: str, state: int = 20) -> WeblateUnitSchema:
    return WeblateUnitSchema(
        id=unit_id,
        language_code="zh_Hans",
        source=source,
        target=target,
        context="",
        state=state,
    )


class FakeWeblate:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.responses: list[list[WeblateUnitSchema]] = []
        self.created: list[str] = []
        self.uploaded: list[CorpusUnitSchema] = []
        self.fail_upload = False

    async def list_units(
        self, slug: str, language: str, q: str = ""
    ) -> list[WeblateUnitSchema]:
        self.queries.append(q)
        return self.responses.pop(0)

    async def create_unit(self, slug: str, draft) -> None:
        self.created.append(draft.source)

    async def upload_targets(
        self, slug: str, language: str, units: list[CorpusUnitSchema]
    ) -> None:
        if self.fail_upload:
            raise WeblateAPIError(502, "upload")
        self.uploaded.extend(units)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncWeblateClient, FakeWeblate]:
    client = AsyncWeblateClient(
        WeblateConfigSchema(url="http://weblate", token="t", project_slug="p")
    )
    weblate = FakeWeblate()
    for name in ("list_units", "create_unit", "upload_targets"):
        monkeypatch.setattr(client, name, getattr(weblate, name))
    return client, weblate


def store(client: AsyncWeblateClient, root: Path) -> LocalGlossaryStore:
    return LocalGlossaryStore(client, root, refresh_seconds=600)


async def test_full_sync_then_incremental_merge(fake, tmp_path: Path) -> None:
    client, weblate = fake
    weblate.responses = [[unit(1, "Frag Grenade", "破片榴弹"), unit(2, "Old", "旧")]]
    first = await store(client, tmp_path).units("base", "zh_Hans")
    assert weblate.queries == ["state:translated"]
    assert len(first) == 2

    weblate.responses = [
        [unit(3, "Acid Grenade", "酸液榴弹"), unit(2, "Old", "", state=0)]
    ]
    second = await store(client, tmp_path).units("base", "zh_Hans")
    assert weblate.queries[1].startswith('changed:>="')
    assert sorted(u.source for u in second) == ["Acid Grenade", "Frag Grenade"]


async def test_stale_snapshot_triggers_full_sync(fake, tmp_path: Path) -> None:
    client, weblate = fake
    old = datetime.now(UTC) - timedelta(days=2)
    (tmp_path / "base.zh_Hans.json").write_text(
        StoredGlossarySchema(
            slug="base", language="zh_Hans", full_synced_at=old, synced_at=old
        ).model_dump_json()
    )
    weblate.responses = [[unit(1, "Frag Grenade", "破片榴弹")]]
    await store(client, tmp_path).units("base", "zh_Hans")
    assert weblate.queries == ["state:translated"]


async def test_units_are_synced_once_per_refresh_window(fake, tmp_path: Path) -> None:
    client, weblate = fake
    weblate.responses = [[unit(1, "Frag Grenade", "破片榴弹")]]
    local = store(client, tmp_path)
    await local.units("base", "zh_Hans")
    await local.units("base", "zh_Hans")
    assert len(weblate.queries) == 1


async def test_deferred_terms_survive_restart_and_flush(fake, tmp_path: Path) -> None:
    client, weblate = fake
    weblate.responses = [[unit(1, "Venom Grenade", "毒液榴弹")]]
    local = store(client, tmp_path)
    writer = DeferredGlossaryWriter(
        local, component_slug="custom", target_lang="zh_Hans"
    )
    added, skipped = await writer.write(
        [
            GlossaryTerm(source="Venom Grenade", target="毒液榴弹", category="item"),
            GlossaryTerm(source="Smoke Grenade", target="烟雾榴弹", category="item"),
        ]
    )
    assert (added, skipped) == (1, 1)
    assert weblate.created == []  # nothing reaches Weblate before flush
    served = await local.units("custom", "zh_Hans")
    assert [u.id for u in served if u.source == "Smoke Grenade"] == [-1]

    restarted = store(client, tmp_path)
    assert restarted.pending("custom", "zh_Hans") == [("Smoke Grenade", "烟雾榴弹")]
    published = await DeferredGlossaryWriter(
        restarted, component_slug="custom", target_lang="zh_Hans"
    ).flush(client)
    assert published == 1
    assert weblate.created == ["Smoke Grenade"]
    assert [u.target for u in weblate.uploaded] == ["烟雾榴弹"]
    assert not (tmp_path / "custom.zh_Hans.pending.jsonl").exists()


async def test_failed_flush_keeps_the_queue(fake, tmp_path: Path) -> None:
    client, weblate = fake
    local = store(client, tmp_path)
    local.add_pending("custom", "zh_Hans", [("Smoke Grenade", "烟雾榴弹")])
    weblate.fail_upload = True
    writer = DeferredGlossaryWriter(
        local, component_slug="custom", target_lang="zh_Hans"
    )
    with pytest.raises(WeblateAPIError):
        await writer.flush(client)
    lines = (tmp_path / "custom.zh_Hans.pending.jsonl").read_text().splitlines()
    assert [json.loads(line)["source"] for line in lines] == ["Smoke Grenade"]
