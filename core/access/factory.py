"""Builds the access layer from the pack's ``access`` section."""

from dataclasses import dataclass

from core.access.auth import Authenticator, JwtAuthenticator, OpenAccess
from core.access.policy import OpenPolicy, Policy, RolePolicy
from core.config import DomainPack


@dataclass(frozen=True)
class AccessControl:
    authenticator: Authenticator
    policy: Policy


def build_access(pack: DomainPack) -> AccessControl:
    config = pack.config.access
    if config is None:
        return AccessControl(OpenAccess(), OpenPolicy())
    return AccessControl(
        JwtAuthenticator(config.jwt, pack.root), RolePolicy.load(pack.root / config.policy)
    )
