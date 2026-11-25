"""Tokens for demo users, signed with the pack's HS256 secret.

Only for local development and evals: in production the company's backend issues tokens.
"""

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jwt
import yaml

from core.access.errors import AccessConfigError
from core.config import AccessConfig, JwtAlgorithm


def load_dev_users(path: Path) -> dict[str, dict[str, Any]]:
    """``users: {handle: {sub: ..., roles: [...], ...}}``: handle -> claims."""
    if not path.is_file():
        raise AccessConfigError(f"dev users file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    users = raw.get("users")
    if not isinstance(users, dict) or not all(
        isinstance(c, dict) and "sub" in c for c in users.values()
    ):
        raise AccessConfigError(f"{path}: expected 'users' mapping handles to claims with 'sub'")
    return users


class DevTokenIssuer:
    def __init__(self, config: AccessConfig, pack_root: Path) -> None:
        jwt_config = config.jwt
        if jwt_config.algorithm is not JwtAlgorithm.HS256 or not config.dev_users:
            raise AccessConfigError("dev tokens need HS256 and a dev_users file in the pack")
        assert jwt_config.secret_env is not None
        self._secret = os.environ.get(jwt_config.secret_env, "")
        if len(self._secret) < 32:
            raise AccessConfigError(f"set {jwt_config.secret_env} (at least 32 characters)")
        self._config = jwt_config
        self.users = load_dev_users(pack_root / config.dev_users)

    def issue(self, handle: str, ttl_seconds: int = 3600) -> str:
        if handle not in self.users:
            raise AccessConfigError(f"no dev user '{handle}'; known: {sorted(self.users)}")
        return self.sign(self.users[handle], ttl_seconds)

    def sign(self, claims: Mapping[str, Any], ttl_seconds: int = 3600) -> str:
        now = int(time.time())
        payload: dict[str, Any] = {**claims, "iat": now, "exp": now + ttl_seconds}
        if self._config.issuer:
            payload["iss"] = self._config.issuer
        if self._config.audience:
            payload["aud"] = self._config.audience
        return jwt.encode(payload, self._secret, algorithm="HS256")
