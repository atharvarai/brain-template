"""Errors of the tools layer."""


class ToolError(Exception):
    """Tool set-up is wrong, for example two servers offer a tool with the same name.

    Failures while running a tool are not raised: they go back to the model as an error
    result, so it can tell the user or try something else.
    """
