"""A deliberately tiny template language: ``{{name}}`` placeholders and nothing else.

No loops, no conditions, no expressions: logic belongs in code, where it is tested.
Rendering is strict in both directions (a missing variable and an unused variable are both
errors) and single-pass, so text inserted for one placeholder is never scanned for more.
"""

import html
import re
from collections.abc import Mapping

from core.prompts.errors import PromptError

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


def placeholders(template: str) -> frozenset[str]:
    """Names of every placeholder in ``template``."""
    return frozenset(_PLACEHOLDER.findall(template))


def render(template: str, variables: Mapping[str, str]) -> str:
    needed = placeholders(template)
    missing = sorted(needed - variables.keys())
    unused = sorted(variables.keys() - needed)
    if missing:
        raise PromptError(f"missing template variable(s): {missing}")
    if unused:
        raise PromptError(f"unused template variable(s): {unused}")
    return _PLACEHOLDER.sub(lambda match: variables[match.group(1)], template)


def wrap_untrusted(tag: str, text: str, **attributes: str | int) -> str:
    """Wrap text we do not control (user input, documents, tool output) in XML-style tags.

    Prompts tell the model that content inside such tags is data, not instructions. The
    closing tag is neutralised inside ``text`` so the content cannot break out of its
    wrapper, and attribute values are escaped for the same reason.
    """
    closing = re.compile(rf"<\s*/\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    rendered = "".join(
        f' {name}="{html.escape(str(value), quote=True)}"' for name, value in attributes.items()
    )
    return f"<{tag}{rendered}>\n{closing.sub(f'[/{tag}]', text)}\n</{tag}>"
