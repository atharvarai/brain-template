"""Structured output: get a validated Pydantic object out of any provider.

The schema is sent through each vendor's native structured-output feature, and the reply is
still validated here. Vendors do not enforce every constraint (lengths, ranges), and some
providers do not enforce the schema at all, so validation plus one corrective retry is what
makes the result trustworthy.
"""

import re
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ValidationError

from core.llm.errors import StructuredOutputError
from core.llm.provider import LLMProvider
from core.llm.types import Completion, LLMRequest, Message, Role, StopReason

# Keywords that vendor "strict" modes reject. Pydantic still enforces them after the call.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "default", "title", "pattern", "format",
        "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems",
    }
)  # fmt: skip
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def strict_json_schema(output_type: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for ``output_type`` in the common subset all vendors accept.

    Every object forbids extra keys and requires all its properties, and constraint
    keywords the vendors reject are removed.
    """
    return _strict(output_type.model_json_schema())  # type: ignore[no-any-return]


def _strict(node: Any, is_property_map: bool = False) -> Any:
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node
    if is_property_map:  # keys here are field names, not schema keywords
        return {name: _strict(schema) for name, schema in node.items()}
    cleaned = {
        key: _strict(value, is_property_map=key in {"properties", "$defs"})
        for key, value in node.items()
        if key not in _UNSUPPORTED_KEYWORDS
    }
    if cleaned.get("type") == "object":
        cleaned["additionalProperties"] = False
        cleaned["required"] = list(cleaned.get("properties", {}))
    return cleaned


async def complete_structured[T: BaseModel](
    provider: LLMProvider, request: LLMRequest, output_type: type[T], max_attempts: int = 2
) -> tuple[T, Completion]:
    """Ask ``provider`` for an ``output_type``. Returns the object and the last completion.

    If the reply fails validation, the model is shown the errors and asked again, up to
    ``max_attempts`` in total. Raises ``StructuredOutputError`` when that is not enough.
    """
    attempt = replace(request, json_schema=strict_json_schema(output_type))
    problem = "no attempt was made"
    for _ in range(max_attempts):
        completion = await provider.complete(attempt)
        if completion.stop_reason in {StopReason.MAX_TOKENS, StopReason.REFUSAL}:
            # Asking again would hit the same wall: surface it instead.
            raise StructuredOutputError(
                f"{provider.name}: no structured output ({completion.stop_reason.value})"
            )
        try:
            return output_type.model_validate_json(_strip_fence(completion.text)), completion
        except ValidationError as exc:
            problem = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}"
                for error in exc.errors()
            )
            attempt = replace(
                attempt,
                messages=(
                    *attempt.messages,
                    Message(Role.ASSISTANT, completion.text or "(empty reply)"),
                    Message(
                        Role.USER,
                        "That reply did not match the required JSON schema: "
                        f"{problem}. Reply again with only the corrected JSON.",
                    ),
                ),
            )
    raise StructuredOutputError(
        f"{provider.name}: output still invalid after {max_attempts} attempts: {problem}"
    )


def _strip_fence(text: str) -> str:
    """Models without enforced output sometimes wrap JSON in a Markdown code fence."""
    stripped = text.strip()
    match = _CODE_FENCE.match(stripped)
    return match.group(1) if match else stripped
