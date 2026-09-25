from src.agent.tools import lookup_glossary_or_patterns, strip_html
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
        results = lookup_glossary_or_patterns(
            "Conventional Weapons Research", self.cache
        )
        assert results[0].source == "Conventional Weapons"

    def test_no_match(self):
        assert lookup_glossary_or_patterns("XY", self.cache) == []

    def test_limit(self):
        assert len(lookup_glossary_or_patterns("Weapons", self.cache, limit=2)) <= 2

    def test_single_word_match(self):
        results = lookup_glossary_or_patterns("Acid Grenade", self.cache)
        assert any(r.source == "Acid" for r in results)
