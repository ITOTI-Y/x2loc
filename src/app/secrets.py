"""Secrets kept in the OS credential store, never in the settings file."""

from contextlib import suppress
from typing import Final, Literal, get_args

import keyring
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

SERVICE: Final = "x2loc"

type SecretName = Literal["llm_api_key", "weblate_token"]

SECRET_NAMES: Final[tuple[SecretName, ...]] = get_args(SecretName.__value__)


class Secrets:
    """Read and write the app's secrets; `backend` defaults to the OS store."""

    def __init__(self, backend: KeyringBackend | None = None) -> None:
        self._backend = backend if backend is not None else keyring.get_keyring()

    def get(self, name: SecretName) -> str:
        return self._backend.get_password(SERVICE, name) or ""

    def set(self, name: SecretName, value: str) -> None:
        self._backend.set_password(SERVICE, name, value)

    def delete(self, name: SecretName) -> None:
        # Already absent is the outcome the caller asked for.
        with suppress(PasswordDeleteError):
            self._backend.delete_password(SERVICE, name)

    def presence(self) -> dict[SecretName, bool]:
        """Which secrets are set, without their values."""
        return {name: bool(self.get(name)) for name in SECRET_NAMES}
