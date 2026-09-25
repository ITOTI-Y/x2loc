from src.core.placeholders import extract_tags, validate_tags


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
