"""Shared test fixtures with auto-generated binary fixture files."""

from collections.abc import Callable
from pathlib import Path

import pytest

from src.core.aligner import BilingualAligner
from src.core.extractor import TermExtractor
from src.core.parser import LocFileParser

FIXTURES_DIR = Path(__file__).parent / "fixtures"

BOM_MAP: dict[str, bytes] = {
    "utf-16-le": b"\xff\xfe",
    "utf-16-be": b"\xfe\xff",
    "utf-8-sig": b"\xef\xbb\xbf",
    "utf-8": b"",
}


def _write_loc_file(path: Path, content: str, encoding: str = "utf-16-le") -> None:
    """Write a localization file with proper BOM prefix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bom = BOM_MAP[encoding]
    raw_encoding = encoding.replace("-sig", "")
    path.write_bytes(bom + content.encode(raw_encoding))


def _generate_fixtures(d: Path) -> None:
    """Generate all binary fixture files if not already present."""
    # --- sample.int (UTF-16-LE) ---
    sample_int_lines = [
        "; File header comment",
        "; Another header comment",
        "",
        "[UIUtilities_Text]",
        "; UI section comment",
        'm_strGenericOK="OK"',
        'm_strGenericCancel="CANCEL"',
        "",
        "[BattleScanner X2AbilityTemplate]",
        'LocFriendlyName="Battle Scanner"',
        'LocHelpText="When thrown, provides vision for '
        '<Ability:BATTLESCANNERDURATION/> turns."',
        'LocLongDescription="<Bullet/> Provides vision in a large area.'
        '<br/><Bullet/> Lasts <Ability:BATTLESCANNERDURATION/> turns."',
        "",
        "[XComGame.UIFinalShell]",
        'm_strStartGame="START GAME"',
        'm_strOptions="OPTIONS"',
        "",
        "[Archetypes.ARC_AdventSecureDoor_x1 XComInteractiveLevelActor]",
        'FriendlyDisplayName="Secure Door"',
        "",
        "[X2ExperienceConfig]",
        'RankNames[0]="Rookie"',
        'RankNames[1]="Squaddie"',
        'RankNames[2]="Corporal"',
        "CharStatLabels[eStat_HP]=HP",
        "",
        "[X2TacticalGameRulesetDataStructures]",
        "; Stat labels",
        'CharStatLabels[eStat_Offense]="AIM"',
        "",
        "[MissionObjectiveTexts]",
        "+MissionDescriptions="
        '(MissionFamily="Recover_LW", '
        'Description="We have discovered '
        '<XGParam:StrValue0/!WorldRegionName/>", MissionIndex=0)',
        "",
        "[UIFinalShell]",
        "; LWOTC Needs Translation",
        'm_strStartButton="%A BEGIN"',
        "",
        "[PlaceholderShowcase]",
        'xgparam_test="Value is <XGParam:IntValue0/!Amount/> units"',
        'ability_test="Damage +<Ability:YOURABILITY/>"',
        'bullet_test="<Bullet/> First point"',
        'heal_test="Heals <Heal/> HP"',
        'br_test="Line one<br/>Line two"',
        "html_test=\"<font color='#33FF33'>green text</font>\"",
        'percent_wrapped_test="Welcome, %PlayerName%!"',
        'percent_test="%A BEGIN HACK"',
        'newline_test="Line1\\nLine2"',
    ]
    _write_loc_file(d / "sample.int", "\n".join(sample_int_lines))

    # --- sample.chn (UTF-16-LE) ---
    chn_lines = [
        "[UIUtilities_Text]",
        'm_strGenericOK="确定"',
        'm_strGenericCancel="取消"',
        "",
        "[BattleScanner X2AbilityTemplate]",
        'LocFriendlyName="战场扫描器"',
    ]
    _write_loc_file(d / "sample.chn", "\n".join(chn_lines))

    # --- empty.int (UTF-16-LE, BOM only) ---
    _write_loc_file(d / "empty.int", "")

    # --- comments_only.int ---
    comments_lines = [
        "; Just a comment",
        "; Another comment",
    ]
    _write_loc_file(d / "comments_only.int", "\n".join(comments_lines))

    # --- entry_before_section.int ---
    bad_lines = [
        'Key="Value"',
        "[Section]",
        'Other="data"',
    ]
    _write_loc_file(d / "entry_before_section.int", "\n".join(bad_lines))

    # --- align_source.int (for alignment tests) ---
    align_src_lines = [
        "[UIUtilities_Text]",
        'm_strGenericOK="OK"',
        'm_strGenericCancel="CANCEL"',
        'm_strSourceOnly="Source Only Text"',
        "",
        "[BattleScanner X2AbilityTemplate]",
        'LocFriendlyName="Battle Scanner"',
        'LocHelpText="Provides vision for <Ability:BATTLESCANNERDURATION/> turns."',
        "",
        "[MissionObjectiveTexts]",
        '+MissionDescriptions=(MissionFamily="Recover_LW", '
        'Description="We have discovered a supply cache.", MissionIndex=0)',
        '+MissionDescriptions=(MissionFamily="Destroy_LW", '
        'Description="We must destroy the relay.", MissionIndex=1)',
        '+MissionDescriptions=(MissionFamily="Rescue_LW", '
        'Description="Rescue the VIP.", MissionIndex=2)',
    ]
    _write_loc_file(d / "align_source.int", "\n".join(align_src_lines))

    # --- align_target.chn (for alignment tests) ---
    align_tgt_lines = [
        "[UIUtilities_Text]",
        'm_strGenericOK="确定"',
        'm_strGenericCancel="取消"',
        "",
        "[BattleScanner X2AbilityTemplate]",
        'LocFriendlyName="战场扫描器"',
        'LocHelpText="提供视野持续<Ability:BATTLESCANNERDURATION/>回合。"',
        'LocTargetOnly="仅目标语言条目"',
        "",
        "[MissionObjectiveTexts]",
        '+MissionDescriptions=(MissionFamily="Recover_LW", '
        'Description="我们发现了一个补给缓存。", MissionIndex=0)',
        '+MissionDescriptions=(MissionFamily="Destroy_LW", '
        'Description="我们必须摧毁中继器。", MissionIndex=1)',
    ]
    _write_loc_file(d / "align_target.chn", "\n".join(align_tgt_lines))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(exist_ok=True)
    _generate_fixtures(FIXTURES_DIR)
    return FIXTURES_DIR


@pytest.fixture
def parser() -> LocFileParser:
    return LocFileParser()


@pytest.fixture
def aligner() -> BilingualAligner:
    return BilingualAligner()


@pytest.fixture
def extractor() -> TermExtractor:
    return TermExtractor()


@pytest.fixture
def sample_int(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.int"


@pytest.fixture
def sample_chn(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.chn"


@pytest.fixture
def empty_int(fixtures_dir: Path) -> Path:
    return fixtures_dir / "empty.int"


@pytest.fixture
def comments_only_int(fixtures_dir: Path) -> Path:
    return fixtures_dir / "comments_only.int"


@pytest.fixture
def entry_before_section_int(fixtures_dir: Path) -> Path:
    return fixtures_dir / "entry_before_section.int"


@pytest.fixture
def align_source_int(fixtures_dir: Path) -> Path:
    return fixtures_dir / "align_source.int"


@pytest.fixture
def align_target_chn(fixtures_dir: Path) -> Path:
    return fixtures_dir / "align_target.chn"


@pytest.fixture
def make_loc_file(tmp_path: Path) -> Callable[..., Path]:
    """Factory fixture: create a temp localization file with given content."""

    def _make(
        content: str,
        filename: str = "test.int",
        encoding: str = "utf-16-le",
    ) -> Path:
        p = tmp_path / filename
        _write_loc_file(p, content, encoding)
        return p

    return _make
