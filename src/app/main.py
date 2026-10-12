"""Start the local app: serve its API and pages on loopback, open a browser."""

import secrets
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, Final

import typer
import uvicorn
from fastapi.staticfiles import StaticFiles
from keyring.backend import KeyringBackend
from loguru import logger

from src.app.api import create_app
from src.app.secrets import Secrets
from src.app.settings import app_home

ROOT: Final = Path(__file__).resolve().parents[2]
WEBUI: Final = Path(__file__).resolve().parents[1] / "webui" / "dist"
WEAPON_NAMES_BUNDLE: Final = ROOT / "mods" / "x2locWeaponCatNames"
HOST: Final = "127.0.0.1"


def credential_store() -> KeyringBackend | None:
    """The Windows Credential Manager on Windows; elsewhere keyring's choice.

    Named outright because keyring finds its backends through package entry
    points, which a compiled executable does not carry.
    """
    if sys.platform != "win32":
        return None
    from keyring.backends.Windows import WinVaultKeyring

    return WinVaultKeyring()


def configure_logging(home: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} {level: <8} {message}")
    logger.add(
        home / "logs" / "x2loc-app.log",
        level="INFO",
        rotation="10 MB",
        retention=3,
        encoding="utf-8",
    )


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def main(
    port: Annotated[int, typer.Option(help="0 picks a free port")] = 0,
    token: Annotated[
        str, typer.Option(help="Session token; random when empty, set for UI work")
    ] = "",
    browser: Annotated[bool, typer.Option(help="Open the app in a browser")] = True,
) -> None:
    home = app_home()
    configure_logging(home)
    session_token = token or secrets.token_urlsafe(24)
    port = port or free_port()
    app = create_app(
        home=home,
        session_token=session_token,
        secrets=Secrets(credential_store()),
        bundle=WEAPON_NAMES_BUNDLE,
    )
    if WEBUI.is_dir():
        app.mount("/", StaticFiles(directory=WEBUI, html=True), name="webui")
    # The token rides in the fragment, which browsers never send to servers.
    url = f"http://{HOST}:{port}/#token={session_token}"
    typer.echo(f"x2loc is running at {url}")
    typer.echo("Keep this window open while you use it; close it to stop x2loc.")
    if browser:
        webbrowser.open(url)
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def run() -> None:
    typer.run(main)
