"""Who is asking."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class User:
    """A verified user. Roles mean nothing to the core: the pack's policy gives them meaning."""

    id: str
    roles: frozenset[str] = frozenset()
    claims: Mapping[str, Any] = field(default_factory=dict)


ANONYMOUS = User("anonymous")


@dataclass(frozen=True, slots=True)
class Caller:
    """The user behind a request, plus the token they came with. The token is forwarded to
    tool servers so they check permissions themselves; the brain never stores it."""

    user: User
    token: str | None = None


ANONYMOUS_CALLER = Caller(ANONYMOUS)
