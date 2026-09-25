from src.agent._share import PATTERN_MAX_EXAMPLES, PATTERN_MAX_SOURCE_WORDS
from src.agent.nodes.pattern_extractor import _detect_patterns, mine_glossary_patterns
from src.models.weblate import WeblateUnitSchema


def test_detects_shared_affix_template() -> None:
    pairs = {
        "Gain 5 Armor": "获得 5 点护甲",
        "Gain 3 Armor": "获得 3 点护甲",
        "Gain 12 Armor": "获得 12 点护甲",
    }
    found = _detect_patterns(pairs)
    assert "Gain {X} Armor" in found
    pattern = found["Gain {X} Armor"]
    assert pattern.tgt_pattern == "获得 {X} 点护甲"
    assert pattern.example_count == 3
    assert len(pattern.examples) == 3


def test_closure_drops_less_specific_variants() -> None:
    pairs = {
        "Gain 5 Armor": "获得 5 点护甲",
        "Gain 3 Armor": "获得 3 点护甲",
        "Gain 12 Armor": "获得 12 点护甲",
    }
    assert set(_detect_patterns(pairs)) == {"Gain {X} Armor"}


def test_requires_min_examples() -> None:
    pairs = {
        "Gain 5 Armor": "获得 5 点护甲",
        "Gain 3 Armor": "获得 3 点护甲",
    }
    assert _detect_patterns(pairs) == {}


def test_rejects_constant_target() -> None:
    pairs = {
        "Reload 1": "装填",
        "Reload 2": "装填",
        "Reload 3": "装填",
    }
    assert _detect_patterns(pairs) == {}


def test_rejects_targets_without_shared_affix() -> None:
    pairs = {
        "Equip 1": "甲",
        "Equip 2": "乙",
        "Equip 3": "丙",
    }
    assert _detect_patterns(pairs) == {}


def test_ignores_single_word_sources() -> None:
    pairs = {"Armor": "护甲", "Shield": "护盾", "Ammo": "弹药"}
    assert _detect_patterns(pairs) == {}


def test_ignores_overlong_sources() -> None:
    padding = " ".join(["pad"] * PATTERN_MAX_SOURCE_WORDS)
    pairs = {f"Prefix {padding} {n}": f"译文 {n}" for n in (1, 2, 3)}
    assert _detect_patterns(pairs) == {}


def test_rejects_markup_opening_prefix() -> None:
    pairs: dict[str, str] = {
        f"<font color='#c9440c'>{en}</font>": f"<font color='#c9440c'>{cn}</font>"
        for en, cn in (("Alpha", "阿尔法"), ("Beta", "贝塔"), ("Gamma", "伽马"))
    }
    assert _detect_patterns(pairs) == {}


def test_rejects_markup_opening_suffix() -> None:
    # "Combo {X}" would map to "连击 {X} <br/>": the tag sits in the source
    # slot but in the target literal, so applying it duplicates the <br/>.
    pairs = {f"Combo {n} <br/>": f"连击 {n} <br/>" for n in (1, 2, 3)}
    assert _detect_patterns(pairs) == {}


def test_examples_capped_at_max() -> None:
    pairs = {f"Gain {n} Armor": f"获得 {n} 点护甲" for n in range(1, 8)}
    pattern = _detect_patterns(pairs)["Gain {X} Armor"]
    assert pattern.example_count == 7
    assert len(pattern.examples) == PATTERN_MAX_EXAMPLES


def test_rejects_template_cutting_through_markup() -> None:
    pairs = {
        "<font color='#114c81'>Resistance to Acid</font>": "<font color='#114c81'>酸液抗性</font>",
        "<font color='#18FF2E'>Resistance to Acid</font>": "<font color='#18FF2E'>酸液抗性</font>",
        "<font color='#DF07B7'>Vulnerability to Acid</font>": "<font color='#DF07B7'>酸液弱点</font>",
    }
    assert _detect_patterns(pairs) == {}


def _glossary(pairs: dict[str, str]) -> dict[str, tuple[WeblateUnitSchema, ...]]:
    return {
        source: (
            WeblateUnitSchema(
                id=i, language_code="zh_Hans", source=source, target=target, context=""
            ),
        )
        for i, (source, target) in enumerate(pairs.items())
    }


def test_glossary_mining_prefers_base_on_conflict() -> None:
    base = _glossary(
        {
            "Laser Rifles": "激光步枪",
            "Coil Rifles": "线圈步枪",
            "Gauss Rifles": "高斯步枪",
        }
    )
    mods = _glossary(
        {"Rail Rifles": "轨道枪", "Beam Rifles": "光束枪", "Ion Rifles": "离子枪"}
    )
    found = mine_glossary_patterns(base, mods)
    assert found["{X} Rifles"][0].tgt_pattern == "{X}步枪"


def test_glossary_mining_combines_sparse_glossaries() -> None:
    base = _glossary({"Frag Grenade": "破片榴弹", "Acid Grenade": "酸液榴弹"})
    mods = _glossary({"Tesla Grenade": "特斯拉榴弹", "Venom Grenade": "毒液榴弹"})
    found = mine_glossary_patterns(base, mods)
    assert found["{X} Grenade"][0].tgt_pattern == "{X}榴弹"
    assert found["{X} Grenade"][0].example_count == 4
