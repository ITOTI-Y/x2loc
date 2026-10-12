"""Publish x2locWeaponCatNames to the Steam Workshop with SteamCMD.

uv run python mods/x2locWeaponCatNames/publish.py --steamcmd <steamcmd.exe> --username <Steam account>

SteamCMD asks for the password and the Steam Guard code itself. The first
run creates the item, hidden, and records its id in `workshop/publishedfileid`;
later runs update that item and keep the visibility set on its page. Tags
and required items cannot be set through SteamCMD: set them on the item's
page.
"""

import re
import shutil
import subprocess
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final

import typer

MOD: Final = "x2locWeaponCatNames"
HERE: Final = Path(__file__).parent
WORKSHOP: Final = HERE / "workshop"
ID_FILE: Final = WORKSHOP / "publishedfileid"
XCOM2_APP_ID: Final = "268500"
TITLE: Final = "[WOTC] x2locWeaponCatNames"
# What the game loads, plus the source other mod authors may want to read.
CONTENT: Final = ("Config", "Script", "Src", f"{MOD}.XComMod")


class Visibility(StrEnum):
    PUBLIC = "public"
    FRIENDS = "friends"
    HIDDEN = "hidden"
    UNLISTED = "unlisted"


VISIBILITY_CODES: Final = {
    Visibility.PUBLIC: 0,
    Visibility.FRIENDS: 1,
    Visibility.HIDDEN: 2,
    Visibility.UNLISTED: 3,
}


def vdf_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def with_published_id(manifest: str, published_id: str) -> str:
    return re.sub(
        r"(?m)^publishedFileId=.*$", f"publishedFileId={published_id}", manifest
    )


def main(
    steamcmd: Annotated[Path, typer.Option(help="Path to steamcmd.exe")],
    username: Annotated[str, typer.Option(help="Steam account that owns the item")],
    changenote: Annotated[
        str, typer.Option(help="Shown in the item's change notes")
    ] = "",
    visibility: Annotated[
        Visibility | None,
        typer.Option(help="Defaults to hidden for a new item, unchanged otherwise"),
    ] = None,
) -> None:
    if not (HERE / "Script" / f"{MOD}.u").is_file():
        raise typer.BadParameter(
            "Script/x2locWeaponCatNames.u is missing: run build.py"
        )
    published_id = (
        ID_FILE.read_text(encoding="utf-8").strip() if ID_FILE.is_file() else "0"
    )
    if published_id == "0" and visibility is None:
        visibility = Visibility.HIDDEN

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / MOD
        stage.mkdir()
        for part in CONTENT:
            source = HERE / part
            if source.is_dir():
                shutil.copytree(source, stage / part)
            else:
                shutil.copy2(source, stage / part)
        manifest = stage / f"{MOD}.XComMod"
        manifest.write_text(
            with_published_id(manifest.read_text(encoding="utf-8"), published_id),
            encoding="utf-8",
        )

        fields = {
            "appid": XCOM2_APP_ID,
            "publishedfileid": published_id,
            "contentfolder": str(stage),
            "previewfile": str(HERE / "ModPreview.jpg"),
            "title": TITLE,
            "description": (WORKSHOP / "description.bbcode").read_text(
                encoding="utf-8"
            ),
            "changenote": changenote,
        }
        if visibility is not None:
            fields["visibility"] = str(VISIBILITY_CODES[visibility])
        vdf = Path(tmp) / "item.vdf"
        body = "\n".join(
            f"\t{vdf_string(k)} {vdf_string(v)}" for k, v in fields.items()
        )
        vdf.write_text(f'"workshopitem"\n{{\n{body}\n}}\n', encoding="utf-8")

        # Interactive: SteamCMD prompts for the password and Steam Guard code.
        subprocess.run(
            [
                str(steamcmd),
                "+login",
                username,
                "+workshop_build_item",
                str(vdf),
                "+quit",
            ],
            check=True,
        )
        # SteamCMD writes the id of a newly created item back into the VDF.
        match = re.search(
            r'"publishedfileid"\s+"(\d+)"', vdf.read_text(encoding="utf-8")
        )

    if match is None or match[1] == "0":
        typer.echo("SteamCMD did not report a published item; see its output above")
        raise typer.Exit(1)
    if match[1] != published_id:
        ID_FILE.write_text(match[1] + "\n", encoding="utf-8")
        source_manifest = HERE / f"{MOD}.XComMod"
        source_manifest.write_text(
            with_published_id(source_manifest.read_text(encoding="utf-8"), match[1]),
            encoding="utf-8",
        )
    typer.echo(f"https://steamcommunity.com/sharedfiles/filedetails/?id={match[1]}")


if __name__ == "__main__":
    typer.run(main)
