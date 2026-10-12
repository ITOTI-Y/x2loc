"""Build the local app into one Windows executable with Nuitka.

Run from the repository root after `pnpm build` in `webui/`:

    uv run --group build python packaging/build.py [--version 1.2.3] [-- extra Nuitka options]

The release workflow runs the same command; locally, without Visual Studio,
pass `-- --zig` to compile with Zig instead of MSVC.
"""

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Annotated, Final

import typer

ROOT: Final = Path(__file__).resolve().parents[1]
WEBUI: Final = Path("src/webui/dist")
WEAPON_NAMES: Final = Path("mods/x2locWeaponCatNames")


def project_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


def main(
    version: Annotated[str, typer.Option(help="Defaults to pyproject's")] = "",
    output_dir: Annotated[Path, typer.Option()] = Path("build/nuitka"),
    extra: Annotated[
        list[str] | None, typer.Argument(help="More Nuitka options")
    ] = None,
) -> None:
    version = (version or project_version()).removeprefix("v")
    if not (ROOT / WEBUI / "index.html").is_file():
        raise typer.BadParameter("build the pages first: cd webui && pnpm build")
    weapon_files = [
        WEAPON_NAMES / "x2locWeaponCatNames.XComMod",
        WEAPON_NAMES / "Script" / "x2locWeaponCatNames.u",
    ]
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "packaging/x2loc_app.py",
        "--mode=onefile",
        f"--output-dir={output_dir}",
        "--output-filename=x2loc.exe",
        "--assume-yes-for-downloads",
        # Packages that load their own submodules by name at run time.
        "--include-package=uvicorn",
        "--include-package=langchain",
        "--include-package=langchain_core",
        "--include-package=langchain_openai",
        "--include-package=langgraph",
        # The app never loads these: test helpers some dependencies ship, and
        # the prompts of the CLI's interactive review.
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=langsmith.testing",
        "--nofollow-import-to=src.ui",
        "--nofollow-import-to=questionary",
        "--nofollow-import-to=prompt_toolkit",
        f"--include-data-dir={WEBUI}={WEBUI}",
        f"--include-data-dir={WEAPON_NAMES / 'Config'}={WEAPON_NAMES / 'Config'}",
        *(f"--include-data-files={path}={path}" for path in weapon_files),
        # Unpack once per version instead of on every start, beside the app's
        # data. A rebuild under an unchanged version reuses the old unpack.
        "--onefile-tempdir-spec={CACHE_DIR}/x2loc/runtime/" + version,
        "--windows-console-mode=force",
        "--product-name=x2loc",
        "--company-name=ITOTI-Y",
        "--file-description=XCOM 2 mod localization",
        f"--product-version={version}",
        f"--file-version={version}",
        *(extra or []),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    typer.run(main)
