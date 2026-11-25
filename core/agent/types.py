"""Values the agent loop works with and the events it yields."""

from dataclasses import dataclass, field

from core.config import ModelSpec
from core.llm import (
    Completion,
    Message,
    StopReason,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


@dataclass(frozen=True, slots=True)
class Budget:
    """Limits for one question. When one runs out, the model answers with what it has."""

    max_steps: int = 6  # model calls
    max_tokens: int = 50_000  # input + output, summed over steps
    max_tool_calls: int = 8


@dataclass(frozen=True, slots=True)
class StepTemplate:
    """What every step sends besides the conversation: the system prompt and the tools."""

    system: str
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int = 2048  # output cap per step


@dataclass(frozen=True, slots=True)
class ToolActivity:
    """One tool run. Streamed as it finishes, and kept on the answer."""

    name: str
    ok: bool
    latency_ms: int
    step: int = 1


@dataclass(frozen=True, slots=True)
class Decision:
    """A person's answer to one action waiting for approval."""

    call_id: str
    approve: bool
    note: str = ""


@dataclass
class RunProgress:
    """Everything the loop has done so far. Mutable, and serialisable, so a run paused for
    approval can be stored and picked up by a later request."""

    messages: list[Message]
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    tool_calls: int = 0
    texts: list[str] = field(default_factory=list)
    activity: list[ToolActivity] = field(default_factory=list)
    spec: ModelSpec | None = None  # the model that answered last
    stop_reason: StopReason = StopReason.END
    # While paused: the calls waiting for approval, results already in hand for the rest of
    # that step's calls, and the order the model asked for them in.
    pending: tuple[ToolCall, ...] = ()
    held: dict[str, ToolResult] = field(default_factory=dict)
    batch: tuple[str, ...] = ()

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, completion: Completion) -> None:
        self.steps += 1
        self.input_tokens += completion.usage.input_tokens
        self.output_tokens += completion.usage.output_tokens
        self.latency_ms += completion.latency_ms
        self.spec, self.stop_reason = completion.spec, completion.stop_reason
        if completion.text:
            self.texts.append(completion.text)

    def completion(self) -> Completion:
        """The whole run as one completion: text joined, usage and latency added up."""
        if self.spec is None:
            raise RuntimeError("no model call has been recorded")
        return Completion(
            text="\n\n".join(self.texts),
            spec=self.spec,
            usage=Usage(self.input_tokens, self.output_tokens),
            stop_reason=self.stop_reason,
            latency_ms=self.latency_ms,
        )


@dataclass(frozen=True, slots=True)
class ApprovalRequired:
    """The run stopped: these calls change data and need a person's go-ahead."""

    pending: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class Finished:
    completion: Completion


AgentEvent = TextDelta | ToolActivity | ApprovalRequired | Finished
