"""Start the local app: serve its API and pages on loopback, open a browser."""

import secrets
import socket
import webbrowser
from pathlib import Path
from typing import Annotated, Final

import typer
import uvicorn
from fastapi.staticfiles import StaticFiles

from src.app.api import create_app
from src.app.secrets import Secrets
from src.app.settings import app_home

ROOT: Final = Path(__file__).resolve().parents[2]
WEBUI: Final = Path(__file__).resolve().parents[1] / "webui" / "dist"
WEAPON_NAMES_BUNDLE: Final = ROOT / "mods" / "x2locWeaponCatNames"
HOST: Final = "127.0.0.1"


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
    session_token = token or secrets.token_urlsafe(24)
    port = port or free_port()
    app = create_app(
        home=app_home(),
        session_token=session_token,
        secrets=Secrets(),
        bundle=WEAPON_NAMES_BUNDLE,
    )
    if WEBUI.is_dir():
        app.mount("/", StaticFiles(directory=WEBUI, html=True), name="webui")
    # The token rides in the fragment, which browsers never send to servers.
    url = f"http://{HOST}:{port}/#token={session_token}"
    typer.echo(f"x2loc is running at {url}")
    if browser:
        webbrowser.open(url)
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def run() -> None:
    typer.run(main)
