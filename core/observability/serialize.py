"""Trace <-> plain dict, for storage and for JSON output."""

from datetime import datetime
from typing import Any

from core.observability.types import Event, Span, Trace, TraceStatus


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "operation": trace.operation,
        "started_at": trace.started_at.isoformat(),
        "duration_ms": trace.duration_ms,
        "status": trace.status.value,
        "error": trace.error,
        "provider": trace.provider,
        "model": trace.model,
        "input_tokens": trace.input_tokens,
        "cache_read_tokens": trace.cache_read_tokens,
        "cache_write_tokens": trace.cache_write_tokens,
        "output_tokens": trace.output_tokens,
        "cost_usd": trace.cost_usd,
        "prompts": list(trace.prompts),
        "retrieved": list(trace.retrieved),
        "cited": list(trace.cited),
        "spans": [
            {
                "name": s.name,
                "offset_ms": s.offset_ms,
                "duration_ms": s.duration_ms,
                "attributes": dict(s.attributes),
            }
            for s in trace.spans
        ],
        "events": [
            {"name": e.name, "offset_ms": e.offset_ms, "attributes": dict(e.attributes)}
            for e in trace.events
        ],
        "attributes": dict(trace.attributes),
        "input": trace.input,
        "output": trace.output,
    }


def trace_from_dict(data: dict[str, Any]) -> Trace:
    return Trace(
        trace_id=data["trace_id"],
        operation=data["operation"],
        started_at=datetime.fromisoformat(data["started_at"]),
        duration_ms=data["duration_ms"],
        status=TraceStatus(data["status"]),
        error=data.get("error"),
        provider=data.get("provider"),
        model=data.get("model"),
        input_tokens=data.get("input_tokens", 0),
        cache_read_tokens=data.get("cache_read_tokens", 0),
        cache_write_tokens=data.get("cache_write_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cost_usd=data.get("cost_usd"),
        prompts=tuple(data.get("prompts", ())),
        retrieved=tuple(data.get("retrieved", ())),
        cited=tuple(data.get("cited", ())),
        spans=tuple(
            Span(s["name"], s["offset_ms"], s["duration_ms"], s.get("attributes", {}))
            for s in data.get("spans", ())
        ),
        events=tuple(
            Event(e["name"], e["offset_ms"], e.get("attributes", {}))
            for e in data.get("events", ())
        ),
        attributes=data.get("attributes", {}),
        input=data.get("input"),
        output=data.get("output"),
    )
