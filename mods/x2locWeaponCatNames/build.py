"""Build x2locWeaponCatNames with the WOTC SDK and install it into the game.

uv run python mods/x2locWeaponCatNames/build.py --sdk <SDK dir> --mods-dir <XComGame/Mods>
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Final

import typer

MOD: Final = "x2locWeaponCatNames"
HERE: Final = Path(__file__).parent


def make(sdk: Path, *args: str) -> None:
    compiler = sdk / "Binaries" / "Win64" / "XComGame.com"
    result = subprocess.run(
        [str(compiler), "make", "-nopause", "-unattended", *args],
        cwd=compiler.parent,
        capture_output=True,
        text=True,
        errors="replace",
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or not any(line.startswith("Success") for line in lines):
        errors = [line for line in lines if "error" in line.lower()]
        raise RuntimeError("SDK make failed:\n" + "\n".join(errors[-20:]))


def main(
    sdk: Annotated[Path, typer.Option(help="XCOM 2 War of the Chosen SDK directory")],
    mods_dir: Annotated[Path, typer.Option(help="The game's XComGame/Mods directory")],
) -> None:
    src = sdk / "Development" / "Src"
    if not src.is_dir():
        # copytree keeps modification times; newer copies would make the
        # compiler rebuild Core.u, which it holds open and cannot replace.
        shutil.copytree(sdk / "Development" / "SrcOrig", src)
    shutil.rmtree(src / MOD, ignore_errors=True)
    shutil.copytree(HERE / "Src" / MOD, src / MOD)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / MOD
        shutil.copytree(HERE / "Config", stage / "Config")
        shutil.copytree(HERE / "Src", stage / "Src")
        shutil.copy2(HERE / f"{MOD}.XComMod", stage)

        make(sdk)
        make(sdk, "-mods", MOD, str(stage))
        (stage / "Script").mkdir()
        shutil.copy2(sdk / "XComGame" / "Script" / f"{MOD}.u", stage / "Script")

        installed = mods_dir / MOD
        shutil.rmtree(installed, ignore_errors=True)
        shutil.copytree(stage, installed)
    typer.echo(f"Installed {installed}")


if __name__ == "__main__":
    typer.run(main)
