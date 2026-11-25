"""JSON form of a paused run, so it survives between requests (and restarts)."""

from typing import Any

from core.agent.types import RunProgress, ToolActivity
from core.config import ModelSpec
from core.llm import Message, Role, StopReason, ToolCall, ToolResult
from core.rag import Chunk


def call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}


def call_from_dict(data: dict[str, Any]) -> ToolCall:
    return ToolCall(data["id"], data["name"], data["arguments"])


def _result_to_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "content": result.content,
        "is_error": result.is_error,
    }


def _result_from_dict(data: dict[str, Any]) -> ToolResult:
    return ToolResult(data["call_id"], data["name"], data["content"], data["is_error"])


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [call_to_dict(c) for c in message.tool_calls],
        "tool_results": [_result_to_dict(r) for r in message.tool_results],
    }


def _message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        Role(data["role"]),
        data["content"],
        tool_calls=tuple(call_from_dict(c) for c in data["tool_calls"]),
        tool_results=tuple(_result_from_dict(r) for r in data["tool_results"]),
    )


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "source": chunk.source,
        "position": chunk.position,
        "title": chunk.title,
        "heading": chunk.heading,
        "text": chunk.text,
        "labels": sorted(chunk.labels),
    }


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        data["source"],
        data["position"],
        data["title"],
        data["heading"],
        data["text"],
        frozenset(data["labels"]),
    )


def progress_to_dict(progress: RunProgress) -> dict[str, Any]:
    return {
        "messages": [_message_to_dict(m) for m in progress.messages],
        "steps": progress.steps,
        "input_tokens": progress.input_tokens,
        "output_tokens": progress.output_tokens,
        "latency_ms": progress.latency_ms,
        "tool_calls": progress.tool_calls,
        "texts": list(progress.texts),
        "activity": [
            {"name": a.name, "ok": a.ok, "latency_ms": a.latency_ms, "step": a.step}
            for a in progress.activity
        ],
        "spec": progress.spec.model_dump(mode="json") if progress.spec else None,
        "stop_reason": progress.stop_reason.value,
        "pending": [call_to_dict(c) for c in progress.pending],
        "held": {cid: _result_to_dict(r) for cid, r in progress.held.items()},
        "batch": list(progress.batch),
    }


def progress_from_dict(data: dict[str, Any]) -> RunProgress:
    return RunProgress(
        messages=[_message_from_dict(m) for m in data["messages"]],
        steps=data["steps"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        latency_ms=data["latency_ms"],
        tool_calls=data["tool_calls"],
        texts=list(data["texts"]),
        activity=[ToolActivity(**a) for a in data["activity"]],
        spec=ModelSpec.model_validate(data["spec"]) if data["spec"] else None,
        stop_reason=StopReason(data["stop_reason"]),
        pending=tuple(call_from_dict(c) for c in data["pending"]),
        held={cid: _result_from_dict(r) for cid, r in data["held"].items()},
        batch=tuple(data["batch"]),
    )
