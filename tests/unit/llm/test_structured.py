import json
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from core.llm import StopReason, StructuredOutputError, complete_structured, strict_json_schema
from core.llm.adapters.fake import FakeProvider
from core.llm.types import Completion, LLMRequest, Usage
from tests.unit.llm.helpers import request, spec

pytestmark = pytest.mark.anyio


class Address(BaseModel):
    city: str
    zip_code: str | None = None


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=3, max_length=60)
    priority: Literal["low", "high"]
    tags: list[str] = []
    address: Address | None = None


def test_strict_schema_is_in_the_subset_every_vendor_accepts() -> None:
    schema = strict_json_schema(Ticket)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["summary", "priority", "tags", "address"]  # all of them
    assert schema["properties"]["summary"] == {"type": "string"}  # length limits removed
    assert schema["properties"]["priority"]["enum"] == ["low", "high"]
    nested = schema["$defs"]["Address"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["city", "zip_code"]
    assert "title" not in json.dumps({k: v for k, v in schema.items() if k != "properties"})


def test_a_field_named_like_a_schema_keyword_is_kept() -> None:
    class Doc(BaseModel):
        title: str
        format: str

    assert set(strict_json_schema(Doc)["properties"]) == {"title", "format"}


async def test_returns_a_validated_object_and_sends_the_schema() -> None:
    provider = FakeProvider(spec(), reply='{"summary": "Refund", "priority": "high", "tags": []}')

    ticket, completion = await complete_structured(provider, request(), Ticket)

    assert ticket == Ticket(summary="Refund", priority="high")
    assert completion.text.startswith("{")
    assert provider.requests[0].json_schema == strict_json_schema(Ticket)


async def test_code_fenced_json_is_accepted() -> None:
    provider = FakeProvider(spec(), reply='```json\n{"summary": "Refund", "priority": "low"}\n```')

    ticket, _ = await complete_structured(provider, request(), Ticket)

    assert ticket.priority == "low"


async def test_invalid_output_is_retried_with_the_validation_errors() -> None:
    provider = FakeProvider(
        spec(),
        replies=['{"summary": "x", "priority": "urgent"}', '{"summary": "Fix", "priority": "low"}'],
    )

    ticket, _ = await complete_structured(provider, request("classify this"), Ticket)

    assert ticket.summary == "Fix"
    retry = provider.requests[1].messages
    assert [m.role.value for m in retry] == ["user", "assistant", "user"]
    assert "priority" in retry[-1].content and "summary" in retry[-1].content


async def test_gives_up_after_max_attempts() -> None:
    provider = FakeProvider(spec(), reply="not json")

    with pytest.raises(StructuredOutputError, match="still invalid after 2 attempts"):
        await complete_structured(provider, request(), Ticket)
    assert len(provider.requests) == 2


@pytest.mark.parametrize("reason", [StopReason.MAX_TOKENS, StopReason.REFUSAL])
async def test_truncation_or_refusal_is_not_retried(reason: StopReason) -> None:
    class Stopped(FakeProvider):
        async def complete(self, request: LLMRequest) -> Completion:
            self.requests.append(request)
            return Completion('{"summ', spec(), Usage(), reason, 0)

    provider = Stopped(spec())

    with pytest.raises(StructuredOutputError, match=reason.value):
        await complete_structured(provider, request(), Ticket)
    assert len(provider.requests) == 1


async def test_fake_provider_fills_any_schema_by_default() -> None:
    class Wide(BaseModel):
        name: str
        count: int
        ok: bool
        level: Literal["a", "b"]
        items: list[str]
        address: Address

    result, _ = await complete_structured(FakeProvider(spec()), request(), Wide)

    assert result.level == "a" and result.address.city == "fake"
