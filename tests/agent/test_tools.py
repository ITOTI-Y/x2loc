from src.agent.tools import lookup_glossary, match_patterns, strip_html
from src.models.agent import PatternSchema
from src.models.weblate import WeblateUnitSchema


class TestStripHtml:
    def test_basic(self):
        assert strip_html("<font color='red'>text</font>") == "text"

    def test_no_html(self):
        assert strip_html("plain text") == "plain text"

    def test_nested(self):
        assert strip_html("<b><i>bold italic</i></b>") == "bold italic"


class TestLookupGlossary:
    def setup_method(self):
        self.cache = {
            "Conventional Weapons": (
                WeblateUnitSchema(
                    id=1,
                    language_code="zh_Hans",
                    source="Conventional Weapons",
                    target="常规武器",
                    context="tech::weapon",
                ),
            ),
            "Magnetic Weapons": (
                WeblateUnitSchema(
                    id=2,
                    language_code="zh_Hans",
                    source="Magnetic Weapons",
                    target="磁力武器",
                    context="tech::weapon",
                ),
            ),
            "Beam Weapons": (
                WeblateUnitSchema(
                    id=3,
                    language_code="zh_Hans",
                    source="Beam Weapons",
                    target="光束武器",
                    context="tech::weapon",
                ),
            ),
            "Acid": (
                WeblateUnitSchema(
                    id=4,
                    language_code="zh_Hans",
                    source="Acid",
                    target="酸液",
                    context="ability::element",
                ),
            ),
        }

    def test_overlap_ranking(self):
        results = lookup_glossary("Conventional Weapons Research", self.cache)
        assert results[0].source == "Conventional Weapons"

    def test_no_match(self):
        assert lookup_glossary("XY", self.cache) == []

    def test_limit(self):
        assert len(lookup_glossary("Weapons", self.cache, limit=2)) <= 2

    def test_single_word_match(self):
        results = lookup_glossary("Acid Grenade", self.cache)
        assert any(r.source == "Acid" for r in results)


def _pattern(src: str, tgt: str, count: int = 3) -> PatternSchema:
    return PatternSchema(
        src_pattern=src, tgt_pattern=tgt, example_count=count, examples=[]
    )


class TestMatchPatterns:
    patterns = {
        p.src_pattern: (p,)
        for p in (
            _pattern("{X} Grenade", "{X}榴弹", 34),
            _pattern("Alien {X}", "外星{X}", 27),
            _pattern("Give {X} Rocket", "给予{X}火箭", 39),
        )
    }

    def test_suffix_template_inside_sentence(self):
        hits = match_patterns("Throws a Plasma Grenade at enemies.", self.patterns)
        assert [p.src_pattern for p in hits] == ["{X} Grenade"]

    def test_prefix_template(self):
        hits = match_patterns("Alien Hunters", self.patterns)
        assert [p.src_pattern for p in hits] == ["Alien {X}"]

    def test_literal_alone_does_not_match(self):
        assert match_patterns("Grenade", self.patterns) == []

    def test_word_boundary(self):
        assert match_patterns("Plasma Grenades", self.patterns) == []

    def test_most_specific_first(self):
        hits = match_patterns("Give Alien Rocket", self.patterns)
        assert [p.src_pattern for p in hits] == ["Give {X} Rocket", "Alien {X}"]
