"""Prompt layer errors. All of them mean a deployment mistake, caught at start-up."""


class PromptError(Exception):
    """A prompt file or template is missing or invalid."""


class PromptNotFoundError(PromptError):
    """No prompt with that name exists in a store."""
