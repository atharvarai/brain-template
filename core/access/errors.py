"""Errors of the access layer."""


class AuthError(Exception):
    """The request's token is missing, malformed, expired or not trusted. Maps to 401."""


class AccessConfigError(Exception):
    """The pack's access set-up is wrong (missing secret, bad policy file)."""
