from src.agent.tools import (
    extract_tags,
    lookup_glossary,
    strip_html,
    tokenize,
    validate_tags,
)


class TestExtractTags:
    def test_html_tags(self):
        assert "<font color='red'>" in extract_tags("<font color='red'>text</font>")
        assert "</font>" in extract_tags("<font color='red'>text</font>")

    def test_printf_format(self):
        assert extract_tags("Deals %d damage") == ["%d"]

    def test_placeholders(self):
        assert extract_tags("Hello {0}, welcome to {Name}") == ["{0}", "{Name}"]

    def test_escape_sequences(self):
        assert extract_tags("Line1\\nLine2\\tEnd") == ["\\n", "\\t"]

    def test_xgparam(self):
        tags = extract_tags("Uses <XGParam:IntValue0/> ammo")
        assert "<XGParam:IntValue0/>" in tags

    def test_no_tags(self):
        assert extract_tags("Simple text") == []

    def test_mixed(self):
        text = "<font color='red'>%d</font> {0} uses \\n"
        assert len(extract_tags(text)) == 5


class TestValidateTags:
    def test_pass(self):
        passed, missing, extra = validate_tags(
            "<font color='red'>%d</font>", "<font color='red'>%d</font>"
        )
        assert passed and not missing and not extra

    def test_missing_tag(self):
        passed, missing, _ = validate_tags(
            "<font color='red'>text</font>", "文本</font>"
        )
        assert not passed
        assert "<font color='red'>" in missing

    def test_extra_tag(self):
        passed, _, extra = validate_tags("text", "<b>文本</b>")
        assert not passed and "<b>" in extra

    def test_no_tags_both(self):
        passed, _, _ = validate_tags("hello", "你好")
        assert passed


class TestStripHtml:
    def test_basic(self):
        assert strip_html("<font color='red'>text</font>") == "text"

    def test_no_html(self):
        assert strip_html("plain text") == "plain text"

    def test_nested(self):
        assert strip_html("<b><i>bold italic</i></b>") == "bold italic"


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Resistance to Acid")
        assert "resistance" in tokens and "acid" in tokens
        assert "to" not in tokens

    def test_strips_html(self):
        tokens = tokenize("<font>Resistance</font>")
        assert "resistance" in tokens and "font" not in tokens


class TestLookupGlossary:
    def setup_method(self):
        self.cache = {
            "Conventional Weapons": {"target": "常规武器", "context": "tech::weapon"},
            "Magnetic Weapons": {"target": "磁力武器", "context": "tech::weapon"},
            "Beam Weapons": {"target": "光束武器", "context": "tech::weapon"},
            "Acid": {"target": "酸液", "context": "ability::element"},
        }

    def test_overlap_ranking(self):
        results = lookup_glossary("Conventional Weapons Research", self.cache)
        assert results[0]["source"] == "Conventional Weapons"

    def test_no_match(self):
        assert lookup_glossary("XY", self.cache) == []

    def test_limit(self):
        assert len(lookup_glossary("Weapons", self.cache, limit=2)) <= 2

    def test_single_word_match(self):
        results = lookup_glossary("Acid Grenade", self.cache)
        assert any(r["source"] == "Acid" for r in results)
