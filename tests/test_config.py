from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import ServiceConfigSchema


def load(tmp_path: Path, *, top: str = "", token: str = "") -> ServiceConfigSchema:
    path = tmp_path / "config.toml"
    path.write_text(
        f'service_token = "s"\ndata_root = "{tmp_path.as_posix()}"\n{top}\n'
        '[steam]\nsteam_username = "u"\nsteam_password = "p"\n'
        f'[weblate]\nurl = "http://weblate"\nproject_slug = "p"\ntoken = "{token}"\n',
        encoding="utf-8",
    )
    return ServiceConfigSchema.from_toml(path)


@pytest.mark.parametrize(
    ("store", "token", "writes"),
    [
        ("auto", "", False),
        ("auto", "wlp_x", True),
        ("local", "wlp_x", False),
        ("weblate", "", True),
    ],
)
def test_store_and_token_decide_where_translations_go(
    tmp_path: Path, store: str, token: str, writes: bool
) -> None:
    config = load(tmp_path, top=f'store = "{store}"', token=token)

    assert config.writes_weblate is writes


def test_priority_must_order_every_source_once(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="priority"):
        load(tmp_path, top='priority = ["manual", "upstream"]')
