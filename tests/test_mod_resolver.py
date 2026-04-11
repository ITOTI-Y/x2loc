"""Tests for mod_resolver walk-up, manifest parsing, and namespace derivation."""

from collections.abc import Callable
from pathlib import Path

import pytest

from src.core.mod_resolver import (
    MAX_WALKUP_LEVELS,
    ModResolveError,
    find_mod_root,
    read_xcommod_title,
    resolve_mod,
    weblate_slug,
)
from src.models.mod import BASE_GAME_NAMESPACE, ModInfoSchema

# ---------------------------------------------------------------------------
# Fixture builders — produce fake mod trees under tmp_path
# ---------------------------------------------------------------------------

ModFactory = Callable[..., Path]


@pytest.fixture
def make_mod(tmp_path: Path) -> ModFactory:
    """Factory fixture that builds a minimal mod directory.

    Returns a function that takes kwargs and produces the mod root path.
    """

    def _make(
        mod_dir: str = "mod-root",
        manifest_name: str = "Mod.XComMod",
        title: str = "Test Mod",
        published_file_id: str = "0",
        loc_subpath: str = "Localization",
        sidecar_id: str | None = None,
        extra_title_prefix: bool = False,
        parent: Path | None = None,
    ) -> Path:
        root = (parent or tmp_path) / mod_dir
        root.mkdir(parents=True)

        manifest_lines = ["[mod]", f"publishedFileId={published_file_id}"]
        if extra_title_prefix:
            # Reproduce the Map[*].XComMod template trap: empty lowercase
            # `title=` before the real `Title=` line.
            manifest_lines.append("title=")
        manifest_lines.append(f"Title={title}")
        manifest_lines.append("RequiresXPACK=true")
        (root / manifest_name).write_text("\n".join(manifest_lines), encoding="utf-8")

        loc_dir = root / loc_subpath
        loc_dir.mkdir(parents=True)
        (loc_dir / "XComGame.int").write_text("", encoding="utf-8")

        if sidecar_id is not None:
            (root / "PublishedFileId.ID").write_text(sidecar_id, encoding="utf-8")

        return root

    return _make


# ---------------------------------------------------------------------------
# find_mod_root
# ---------------------------------------------------------------------------


class TestFindModRoot:
    def test_depth_one_walkup(self, tmp_path: Path, make_mod: ModFactory) -> None:
        """Loc dir is a direct child of mod root (the common case)."""
        mod_root = make_mod()
        found = find_mod_root(mod_root / "Localization", tmp_path)
        assert found == mod_root

    def test_depth_two_submod(self, tmp_path: Path, make_mod: ModFactory) -> None:
        """Loc files live in a submod dir (LWOTC nesting pattern)."""
        mod_root = make_mod(loc_subpath="Localization/SubMod")
        found = find_mod_root(mod_root / "Localization" / "SubMod", tmp_path)
        assert found == mod_root

    def test_depth_three_deepest_observed(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """Three parent hops — matches the deepest real-world case."""
        mod_root = make_mod(loc_subpath="Localization/Group/SubMod")
        start = mod_root / "Localization" / "Group" / "SubMod"
        found = find_mod_root(start, tmp_path)
        assert found == mod_root

    def test_zero_walkup_start_at_root(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """Start path is already the mod root."""
        mod_root = make_mod()
        found = find_mod_root(mod_root, tmp_path)
        assert found == mod_root

    def test_start_outside_sandbox_rejected(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """Walk-up must never escape the sandbox boundary."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside_mod = make_mod(mod_dir="outside", parent=tmp_path)

        with pytest.raises(ModResolveError, match="not inside sandbox"):
            find_mod_root(outside_mod / "Localization", sandbox)

    def test_parent_relative_path_rejected(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """`..` components must be collapsed before the containment check."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside_mod = make_mod(mod_dir="outside", parent=tmp_path)
        escape = sandbox / ".." / outside_mod.name / "Localization"

        with pytest.raises(ModResolveError, match="not inside sandbox"):
            find_mod_root(escape, sandbox)

    def test_no_manifest_raises(self, tmp_path: Path) -> None:
        (tmp_path / "loc").mkdir()
        with pytest.raises(ModResolveError, match=r"No \.XComMod found"):
            find_mod_root(tmp_path / "loc", tmp_path)

    def test_multiple_manifests_rejected(self, tmp_path: Path) -> None:
        mod_root = tmp_path / "mod"
        mod_root.mkdir()
        (mod_root / "First.XComMod").write_text("[mod]\nTitle=A\n")
        (mod_root / "Second.XComMod").write_text("[mod]\nTitle=B\n")
        (mod_root / "Localization").mkdir()

        with pytest.raises(ModResolveError, match=r"Multiple \.XComMod"):
            find_mod_root(mod_root / "Localization", tmp_path)

    def test_max_levels_cap(self, tmp_path: Path) -> None:
        """Walk-up gives up at `max_levels` even within the sandbox."""
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)

        with pytest.raises(ModResolveError, match="max_levels=2"):
            find_mod_root(deep, tmp_path, max_levels=2)

    def test_case_insensitive_manifest_extension(self, tmp_path: Path) -> None:
        mod_root = tmp_path / "mod"
        mod_root.mkdir()
        (mod_root / "Weird.xcommod").write_text("[mod]\nTitle=Weird\n")
        (mod_root / "Localization").mkdir()

        found = find_mod_root(mod_root / "Localization", tmp_path)
        assert found == mod_root


# ---------------------------------------------------------------------------
# read_xcommod_title
# ---------------------------------------------------------------------------


class TestReadXcommodTitle:
    def test_plain_title(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.XComMod"
        manifest.write_text("[mod]\nTitle=More Traits\n")
        assert read_xcommod_title(manifest) == "More Traits"

    def test_template_duplicate_title_trap(self, tmp_path: Path) -> None:
        """Template leaves empty `title=` followed by real `Title=` — the
        parser MUST pick the first non-empty value, not last-write-wins.
        """
        manifest = tmp_path / "m.XComMod"
        manifest.write_text("[mod]\ntitle=\ndescription=\nTitle=Map [Cliffs]\n")
        assert read_xcommod_title(manifest) == "Map [Cliffs]"

    def test_case_insensitive_key(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.XComMod"
        manifest.write_text("[mod]\nTITLE=Shouting\n")
        assert read_xcommod_title(manifest) == "Shouting"

    def test_ignores_keys_outside_mod_section(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.XComMod"
        manifest.write_text("[other]\nTitle=Wrong\n[mod]\nTitle=Right\n")
        assert read_xcommod_title(manifest) == "Right"

    def test_utf8_bom_tolerated(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.XComMod"
        manifest.write_bytes(b"\xef\xbb\xbf[mod]\nTitle=BOMed\n")
        assert read_xcommod_title(manifest) == "BOMed"

    def test_missing_title_raises(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.XComMod"
        manifest.write_text("[mod]\npublishedFileId=0\n")
        with pytest.raises(ModResolveError, match="No non-empty Title"):
            read_xcommod_title(manifest)


# ---------------------------------------------------------------------------
# weblate_slug
# ---------------------------------------------------------------------------


class TestWeblateSlug:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Long War of the Chosen", "long-war-of-the-chosen"),
            ("More Traits", "more-traits"),
            ("[WOTC] Evac All", "wotc-evac-all"),
            ("Map [Cliffs]", "map-cliffs"),
            ("A Harder War: Absurd Aliens", "a-harder-war-absurd-aliens"),
            # Transliteration
            ("中国武器包", "zhong-guo-wu-qi-bao"),
            ("日本語MOD", "ri-ben-yu-mod"),
            # Punctuation collapse
            ("   spaces   and...dots   ", "spaces-and-dots"),
            ("Trailing---dashes---", "trailing-dashes"),
        ],
    )
    def test_common_titles(self, text: str, expected: str) -> None:
        assert weblate_slug(text) == expected

    def test_empty_input_falls_back(self) -> None:
        assert weblate_slug("") == "unnamed"

    def test_pure_punctuation_falls_back(self) -> None:
        assert weblate_slug("!!!---???") == "unnamed"

    def test_truncation_strips_trailing_hyphen(self) -> None:
        # Length 5 truncates "long-title-name" to "long-" — result must not
        # end with a dangling hyphen.
        assert weblate_slug("long title name", max_len=5) == "long"


# ---------------------------------------------------------------------------
# resolve_mod — end-to-end
# ---------------------------------------------------------------------------


class TestResolveMod:
    def test_numeric_dir_name_becomes_steam_id(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """Workshop download layout: directory name = Steam ID."""
        mod_root = make_mod(mod_dir="1122837889", title="More Traits")
        info = resolve_mod(mod_root / "Localization", tmp_path)

        assert info.steam_id == "1122837889"
        assert info.mod_title == "More Traits"
        assert info.namespace == "1122837889-more-traits"
        assert info.mod_root == mod_root

    def test_sidecar_beats_manifest_zero(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """`PublishedFileId.ID` overrides `publishedFileId=0` in manifest."""
        mod_root = make_mod(
            mod_dir="lwotc-src",
            title="Long War of the Chosen",
            published_file_id="0",
            sidecar_id="2985868530",
        )
        info = resolve_mod(mod_root / "Localization", tmp_path)

        assert info.steam_id == "2985868530"
        assert info.namespace == "2985868530-long-war-of-the-chosen"

    def test_non_numeric_dir_and_zero_manifest_degrades(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """Zip-extracted mod with no sidecar and manifest=0 → `local-`."""
        mod_root = make_mod(
            mod_dir="extracted-test-mod",
            title="WC_qUIck_LW2",
            published_file_id="0",
        )
        info = resolve_mod(mod_root / "Localization", tmp_path)

        assert info.steam_id is None
        assert info.namespace == "local-wc-quick-lw2"

    def test_explicit_steam_id_override(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """CLI-supplied Steam ID beats everything (including sidecars)."""
        mod_root = make_mod(
            mod_dir="whatever",
            title="Example",
            published_file_id="0",
            sidecar_id="9999",
        )
        info = resolve_mod(mod_root / "Localization", tmp_path, steam_id_override="42")

        assert info.steam_id == "42"
        assert info.namespace == "42-example"

    def test_manifest_published_file_id_fallback(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        """Non-numeric dir name, no sidecar, but manifest has real ID."""
        mod_root = make_mod(
            mod_dir="my-cool-mod",
            title="My Cool Mod",
            published_file_id="1234567890",
        )
        info = resolve_mod(mod_root / "Localization", tmp_path)

        assert info.steam_id == "1234567890"
        assert info.namespace == "1234567890-my-cool-mod"

    def test_duplicate_title_trap_end_to_end(
        self, tmp_path: Path, make_mod: ModFactory
    ) -> None:
        mod_root = make_mod(
            mod_dir="3388852030",
            title="Map [Office]",
            extra_title_prefix=True,
        )
        info = resolve_mod(mod_root / "Localization", tmp_path)

        assert info.mod_title == "Map [Office]"
        assert info.namespace == "3388852030-map-office"

    def test_unicode_title(self, tmp_path: Path, make_mod: ModFactory) -> None:
        mod_root = make_mod(mod_dir="9000000001", title="中国武器包")
        info = resolve_mod(mod_root / "Localization", tmp_path)

        assert info.namespace == "9000000001-zhong-guo-wu-qi-bao"

    def test_max_walkup_constant_unchanged(self) -> None:
        """Pin the cap — changing it should be a deliberate decision
        backed by rescanning real mod depth distribution."""
        assert MAX_WALKUP_LEVELS == 5


# ---------------------------------------------------------------------------
# Base game special path
# ---------------------------------------------------------------------------


class TestBaseGame:
    def test_base_game_constant_namespace(self) -> None:
        info = ModInfoSchema.base_game()
        assert info.namespace == BASE_GAME_NAMESPACE == "base-xcom2-wotc"
        assert info.steam_id is None
        assert info.mod_root is None
        assert info.xcommod_path is None
        assert "XCOM 2" in info.mod_title
