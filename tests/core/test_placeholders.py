from src.core.placeholders import (
    extract_tags,
    is_malformed,
    repair_markup,
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


BROKEN = "<font color='#df07b7'Smart Magazines</font>"


def test_repair_markup_closes_an_opener_that_lost_its_bracket() -> None:
    assert repair_markup(BROKEN) == "<font color='#df07b7'>Smart Magazines</font>"


def test_repair_markup_leaves_well_formed_and_ambiguous_markup_alone() -> None:
    fine = "<font color='#df07b7'>X</font> <Bullet/> <img src='a.png' width='3'/>"
    assert repair_markup(fine) == fine
    assert repair_markup("a<b c") == "a<b c"


def test_translation_closing_the_tag_is_valid_against_a_broken_source() -> None:
    assert validate_tags(BROKEN, "<font color='#df07b7'>智能弹匣</font>")[0]
    assert not validate_tags(BROKEN, BROKEN)[0]
    assert not validate_tags(BROKEN, "智能弹匣")[0]


def test_unrepairable_source_defers_to_a_well_formed_translation() -> None:
    source = "<font color=#df07b7 Smart</font>"
    assert is_malformed(repair_markup(source))
    assert validate_tags(source, "<font color='#df07b7'>智能</font>")[0]
    assert validate_tags(source, source)[0]
    assert not validate_tags(source, "智能")[0]


def test_verbatim_copy_of_an_unrepairable_source_stays_valid() -> None:
    source = "<Bullet/> Grants 1 hit point.\n<<Bullet/> Reduces infiltration time."
    translation = "<Bullet/> 提供1点生命值。\n<<Bullet/> 缩短渗透时间。"
    assert validate_tags(source, translation)[0]
    assert validate_tags(source, "<Bullet/> 提供。\n<Bullet/> 缩短。")[0]
    assert not validate_tags(source, "<Bullet/> 提供。")[0]
