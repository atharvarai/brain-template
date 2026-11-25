"""Authenticators: turn the bearer token on a request into a ``User``."""

import os
from pathlib import Path
from typing import Protocol

import jwt

from core.access.errors import AccessConfigError, AuthError
from core.access.types import ANONYMOUS, User
from core.config import JwtAlgorithm, JwtConfig


class Authenticator(Protocol):
    @property
    def required(self) -> bool:
        """False when access control is off and every request is anonymous."""
        ...

    def verify(self, token: str | None) -> User:
        """Return the user, or raise ``AuthError``."""
        ...


class OpenAccess:
    """No access control: everyone is anonymous. Used when the pack has no ``access``."""

    required = False

    def verify(self, token: str | None) -> User:
        return ANONYMOUS


class JwtAuthenticator:
    """Checks signature, expiry and, when configured, issuer and audience.

    The token must carry ``sub`` (the user id) and ``exp``. Roles come from the configured
    claim and must be a list of strings.
    """

    required = True

    def __init__(self, config: JwtConfig, pack_root: Path) -> None:
        self._config = config
        self._key = _verification_key(config, pack_root)

    def verify(self, token: str | None) -> User:
        if not token:
            raise AuthError("missing bearer token")
        config = self._config
        try:
            claims = jwt.decode(
                token,
                self._key,
                algorithms=[config.algorithm.value],  # never trust the token's own header
                issuer=config.issuer,
                audience=config.audience,
                leeway=config.leeway_seconds,
                options={"require": ["sub", "exp"], "verify_aud": config.audience is not None},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(f"invalid token: {exc}") from exc
        roles = claims.get(config.roles_claim, [])
        if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
            raise AuthError(f"claim '{config.roles_claim}' must be a list of strings")
        return User(str(claims["sub"]), frozenset(roles), claims)


def _verification_key(config: JwtConfig, pack_root: Path) -> str:
    if config.algorithm is JwtAlgorithm.HS256:
        assert config.secret_env is not None  # the schema guarantees it
        secret = os.environ.get(config.secret_env, "")
        if len(secret) < 32:
            raise AccessConfigError(
                f"set {config.secret_env} to a secret of at least 32 characters "
                "(it signs and checks user tokens)"
            )
        return secret
    assert config.public_key_file is not None
    path = pack_root / config.public_key_file
    if not path.is_file():
        raise AccessConfigError(f"public key file not found: {path}")
    return path.read_text(encoding="utf-8")
