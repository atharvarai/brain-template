"""The pack's access policy: which document labels and which tools each role may use.

``policy.yaml`` in the pack::

    roles:
      member:  {read: [],          tools: [lookup_order]}
      staff:   {read: [internal],  tools: [lookup_order, refund_order]}
      admin:   {read: [internal, finance], tools: ["*"]}

Deny by default: a role not listed, or a user with no roles, reads only unlabelled
documents and calls no tools. A user with several roles gets the union.
"""

from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from core.access.errors import AccessConfigError
from core.access.types import User

ALL_TOOLS = "*"


class Policy(Protocol):
    def allowed_labels(self, user: User) -> frozenset[str]: ...

    def can_call(self, user: User, tool: str) -> bool: ...


class OpenPolicy:
    """No access control: unlabelled documents only, every tool."""

    def allowed_labels(self, user: User) -> frozenset[str]:
        return frozenset()

    def can_call(self, user: User, tool: str) -> bool:
        return True


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RoleRules(_Strict):
    read: tuple[str, ...] = ()  # labels this role may read, besides unlabelled documents
    tools: tuple[str, ...] = ()  # tool names, or "*" for every tool


class PolicyFile(_Strict):
    roles: dict[str, RoleRules]


class RolePolicy:
    def __init__(self, rules: PolicyFile) -> None:
        self._roles = rules.roles

    @classmethod
    def load(cls, path: Path) -> "RolePolicy":
        if not path.is_file():
            raise AccessConfigError(f"policy file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls(PolicyFile.model_validate(raw))
        except (yaml.YAMLError, ValidationError) as exc:
            raise AccessConfigError(f"{path} is invalid:\n{exc}") from exc

    def allowed_labels(self, user: User) -> frozenset[str]:
        return frozenset(label for rules in self._rules(user) for label in rules.read)

    def can_call(self, user: User, tool: str) -> bool:
        return any(ALL_TOOLS in r.tools or tool in r.tools for r in self._rules(user))

    def _rules(self, user: User) -> list[RoleRules]:
        return [self._roles[role] for role in sorted(user.roles) if role in self._roles]
