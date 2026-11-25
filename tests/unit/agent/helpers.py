"""A pipeline wired to a scripted model and in-memory tools, for agent tests."""

from collections.abc import Collection, Sequence
from pathlib import Path

from core.access import AuditLog, Policy
from core.agent import Agent, Budget, InMemoryRunStore
from core.config import ModelSpec, ModelTier, Provider, load_pack
from core.llm import Message, ModelGateway, Role, ToolCall, ToolResult, ToolSpec
from core.llm.adapters.fake import FakeProvider, Reply
from core.observability import InMemoryTraceStore, Tracer
from core.pipeline import Pipeline
from core.prompts import ChainedPromptStore, FilePromptStore, build_system_prompt, builtin_store

EMPTY = {"type": "object", "properties": {}}
FIND = ToolSpec("find_order", "Find an order.", EMPTY)
STOCK = ToolSpec("check_stock", "Check stock.", EMPTY)
CANCEL = ToolSpec("cancel_order", "Cancel an order.", EMPTY)


class FakeTools:
    """``find_order`` and ``check_stock`` are read-only; ``cancel_order`` changes data."""

    def __init__(self, failing: Collection[str] = ()) -> None:
        self.specs = (FIND, STOCK, CANCEL)
        self._failing = set(failing)
        self.ran: list[ToolCall] = []
        self.credentials: list[str | None] = []

    def is_read_only(self, name: str) -> bool:
        return name in {"find_order", "check_stock"}

    async def run(self, call: ToolCall, credentials: str | None = None) -> ToolResult:
        self.ran.append(call)
        self.credentials.append(credentials)
        if call.name in self._failing:
            return ToolResult(call.id, call.name, "service down", is_error=True)
        return ToolResult(call.id, call.name, f"{call.name} result")


def calls(*names: str, prefix: str = "c") -> tuple[ToolCall, ...]:
    return tuple(ToolCall(f"{prefix}{i}", name, {"id": i}) for i, name in enumerate(names))


def build(
    pack_dir: Path,
    replies: Sequence[Reply],
    tools: FakeTools | None = None,
    budget: Budget | None = None,
    auto_approve: Collection[str] = (),
    policy: Policy | None = None,
    audit: AuditLog | None = None,
) -> tuple[Pipeline, FakeProvider, FakeTools, InMemoryTraceStore]:
    pack = load_pack(pack_dir)
    prompts = ChainedPromptStore([FilePromptStore(pack.prompts_dir), builtin_store()])
    provider = FakeProvider(ModelSpec(provider=Provider.FAKE, model="m"), replies=replies)
    tools = tools or FakeTools()
    store = InMemoryTraceStore()
    pipeline = Pipeline(
        ModelGateway({tier: provider for tier in ModelTier}),
        build_system_prompt(prompts, pack.config),
        _NoDocs(),
        prompts.get("rag_turn"),
        Tracer(store),
        agent=Agent(tools, budget, auto_approve, policy, audit),
        runs=InMemoryRunStore(),
        policy=policy,
    )
    return pipeline, provider, tools, store


def ask(text: str = "where is my order?") -> list[Message]:
    return [Message(Role.USER, text)]


class _NoDocs:
    async def search(self, query: str, k: int, allowed_labels: frozenset[str] | None) -> list:
        return []
