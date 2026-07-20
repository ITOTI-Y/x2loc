from src.agent._share import PATTERN_MAX_EXAMPLES, PATTERN_MAX_SOURCE_WORDS
from src.agent.nodes.pattern_extractor import _detect_patterns


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
    assert pattern.approved_count == 3
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
    pairs = {f"Combo {n} <br/>": f"连击 {n} <br/>" for n in (1, 2, 3)}
    found = _detect_patterns(pairs)
    assert "Combo {X} <br/>" not in found
    assert "Combo {X}" in found


def test_examples_capped_at_max() -> None:
    pairs = {f"Gain {n} Armor": f"获得 {n} 点护甲" for n in range(1, 8)}
    pattern = _detect_patterns(pairs)["Gain {X} Armor"]
    assert pattern.approved_count == 7
    assert len(pattern.examples) == PATTERN_MAX_EXAMPLES
