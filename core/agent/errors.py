"""Errors of the agent layer. The API maps them to 404, 409 and 422."""


class RunNotFoundError(Exception):
    """No paused run with that id, or it expired."""


class RunClosedError(Exception):
    """The run was already resumed: an approval is used once, never replayed."""


class DecisionError(ValueError):
    """The decisions do not match the actions waiting for approval."""
