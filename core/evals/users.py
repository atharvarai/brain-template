"""Who each eval case asks as. With access control on, cases run as the pack's demo users,
with real signed tokens, so retrieval filters, tool policy and the tool server's own checks
are all exercised."""

from core.access import ANONYMOUS_CALLER, Authenticator, Caller, DevTokenIssuer
from core.config import DomainPack
from core.evals.cases import EvalCase, EvalError


class EvalUsers:
    def __init__(self, pack: DomainPack, authenticator: Authenticator) -> None:
        access = pack.config.access
        self._default = pack.config.evals.default_user
        self._issuer = DevTokenIssuer(access, pack.root) if access else None
        self._authenticator = authenticator
        self._cache: dict[str, Caller] = {}

    def caller(self, case: EvalCase) -> Caller:
        handle = case.user or self._default
        if self._issuer is None:
            if case.user:
                raise EvalError(f"case {case.id} names a user, but the pack has no access control")
            return ANONYMOUS_CALLER
        if handle is None:
            raise EvalError(f"case {case.id} names no user and evals.default_user is not set")
        if handle not in self._cache:
            if handle not in self._issuer.users:
                raise EvalError(f"case {case.id}: no demo user '{handle}'")
            token = self._issuer.issue(handle)
            # Through the real authenticator, so a token the API would reject fails here too.
            self._cache[handle] = Caller(self._authenticator.verify(token), token)
        return self._cache[handle]
