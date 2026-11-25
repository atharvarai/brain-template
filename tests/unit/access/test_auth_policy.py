"""Tokens, the role policy, dev tokens and the audit log."""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from core.access import (
    AccessConfigError,
    AuditAction,
    AuditEntry,
    AuthError,
    DevTokenIssuer,
    JwtAuthenticator,
    RolePolicy,
    SqliteAuditLog,
    User,
)
from core.access.policy import PolicyFile
from core.config import AccessConfig, JwtConfig
from tests.conftest import TEST_JWT_SECRET

CONFIG = JwtConfig(secret_env="BRAIN_JWT_SECRET", issuer="acme-backend", audience="acme-brain")


def sign(claims: dict, secret: str = TEST_JWT_SECRET, **headers: object) -> str:
    now = int(time.time())
    payload = {"iss": "acme-backend", "aud": "acme-brain", "iat": now, "exp": now + 60, **claims}
    return jwt.encode(payload, secret, algorithm="HS256", headers=headers or None)


def auth(tmp_path: Path, config: JwtConfig = CONFIG) -> JwtAuthenticator:
    return JwtAuthenticator(config, tmp_path)


def test_a_valid_token_becomes_a_user_with_roles(tmp_path: Path) -> None:
    user = auth(tmp_path).verify(sign({"sub": "u1", "roles": ["staff"], "name": "Asha"}))

    assert (user.id, user.roles, user.claims["name"]) == ("u1", frozenset({"staff"}), "Asha")


@pytest.mark.parametrize(
    ("token", "reason"),
    [
        (None, "missing"),
        ("not-a-jwt", "invalid"),
        (sign({"sub": "u1"}, secret="another-secret-that-is-long-enough-000"), "Signature"),
        (sign({"sub": "u1", "exp": int(time.time()) - 600}), "expired"),
        (sign({"sub": "u1", "aud": "someone-else"}), "(?i)audience"),
        (sign({"sub": "u1", "iss": "evil"}), "(?i)issuer"),
        (sign({"roles": ["staff"]}), "sub"),
        (sign({"sub": "u1", "roles": "admin"}), "list of strings"),
    ],
)
def test_bad_tokens_are_refused(tmp_path: Path, token: str | None, reason: str) -> None:
    with pytest.raises(AuthError, match=reason):
        auth(tmp_path).verify(token)


def test_an_unsigned_token_is_refused(tmp_path: Path) -> None:
    unsigned = jwt.encode({"sub": "u1", "exp": int(time.time()) + 60}, None, algorithm="none")
    with pytest.raises(AuthError):
        auth(tmp_path).verify(unsigned)


def test_a_short_or_missing_secret_stops_start_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAIN_JWT_SECRET", "short")
    with pytest.raises(AccessConfigError, match="at least 32"):
        auth(tmp_path)


def test_rs256_tokens_are_checked_with_the_public_key_only(tmp_path: Path) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    (tmp_path / "backend.pem").write_bytes(public)
    config = JwtConfig(algorithm="RS256", public_key_file="backend.pem")
    token = jwt.encode({"sub": "u9", "exp": int(time.time()) + 60}, key, algorithm="RS256")

    assert auth(tmp_path, config).verify(token).id == "u9"
    # Algorithm confusion: HS256 "signed" with the public key as the secret must not pass.
    with pytest.raises(AuthError):
        auth(tmp_path, config).verify(forge_hs256({"sub": "u9", "exp": 9_999_999_999}, public))


def forge_hs256(claims: dict, key: bytes) -> str:
    def part(data: dict) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signing_input = f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part(claims)}"
    mac = hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(mac).rstrip(b'=').decode()}"


# --- policy ---------------------------------------------------------------------------------

POLICY = RolePolicy(
    PolicyFile.model_validate(
        {
            "roles": {
                "member": {"read": [], "tools": ["lookup"]},
                "staff": {"read": ["internal"], "tools": ["lookup", "refund"]},
                "admin": {"read": ["internal", "finance"], "tools": ["*"]},
            }
        }
    )
)


def user(*roles: str) -> User:
    return User("u", frozenset(roles))


def test_roles_grant_labels_and_tools_and_combine() -> None:
    assert POLICY.allowed_labels(user("member")) == frozenset()
    assert POLICY.allowed_labels(user("member", "staff")) == {"internal"}
    assert POLICY.can_call(user("member"), "lookup")
    assert not POLICY.can_call(user("member"), "refund")
    assert POLICY.can_call(user("member", "staff"), "refund")
    assert POLICY.can_call(user("admin"), "anything")


def test_anything_not_granted_is_denied() -> None:
    assert POLICY.allowed_labels(user()) == frozenset()
    assert not POLICY.can_call(user(), "lookup")
    assert not POLICY.can_call(user("visitor"), "lookup")  # a role the policy does not know


def test_a_bad_policy_file_is_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "policy.yaml").write_text("roles: {member: {write: [x]}}")
    with pytest.raises(AccessConfigError, match="invalid"):
        RolePolicy.load(tmp_path / "policy.yaml")
    with pytest.raises(AccessConfigError, match="not found"):
        RolePolicy.load(tmp_path / "missing.yaml")


# --- dev tokens and audit -------------------------------------------------------------------


def test_dev_tokens_pass_the_real_authenticator(tmp_path: Path) -> None:
    (tmp_path / "users.yaml").write_text("users: {asha: {sub: u1, roles: [staff]}}")
    access = AccessConfig(jwt=CONFIG, dev_users="users.yaml")
    issuer = DevTokenIssuer(access, tmp_path)

    verified = auth(tmp_path).verify(issuer.issue("asha"))

    assert (verified.id, verified.roles) == ("u1", frozenset({"staff"}))
    with pytest.raises(AccessConfigError, match="no dev user"):
        issuer.issue("nobody")


def test_the_audit_log_keeps_entries_in_order(tmp_path: Path) -> None:
    log = SqliteAuditLog(tmp_path / "audit.sqlite3")
    asha = User("u1", frozenset({"staff"}))
    log.record(AuditEntry.of(AuditAction.TOOL_CALL, asha, "lookup", "ok", detail={"id": 7}))
    log.record(AuditEntry.of(AuditAction.TOOL_DENIED, asha, "refund", "denied", trace_id="t1"))

    latest, first = log.recent(10)

    assert (latest.action, latest.target, latest.trace_id) == (
        AuditAction.TOOL_DENIED, "refund", "t1",
    )  # fmt: skip
    assert (first.user_id, first.roles, dict(first.detail)) == ("u1", ("staff",), {"id": 7})
