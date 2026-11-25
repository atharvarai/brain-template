"""Access enforced in the request path: what the model is offered, what may run, whose
token goes to the tool server, and who may resume a paused run."""

from pathlib import Path

import pytest
import yaml

from core import cli
from core.access import AuditAction, Caller, InMemoryAuditLog, RolePolicy, User
from core.access.policy import PolicyFile
from core.agent import DENIED, Decision, RunNotFoundError
from core.api.app import build_context
from core.config import Settings
from core.llm import default_factory
from tests.unit.agent.helpers import ask, build, calls

pytestmark = pytest.mark.anyio

POLICY = RolePolicy(
    PolicyFile.model_validate(
        {"roles": {"member": {"tools": ["find_order", "cancel_order"]},
                   "staff": {"read": ["internal"], "tools": ["*"]}}}
    )
)  # fmt: skip
MEMBER = Caller(User("m1", frozenset({"member"})), token="member-token")
OTHER = Caller(User("m2", frozenset({"member"})), token="other-token")


async def test_the_model_is_offered_only_the_tools_the_user_may_call(pack_dir: Path) -> None:
    pipeline, provider, _, _ = build(pack_dir, ["ok"], policy=POLICY)

    await pipeline.answer(ask(), caller=MEMBER)

    assert [t.name for t in provider.requests[0].tools] == ["find_order", "cancel_order"]


async def test_a_call_outside_the_policy_is_refused_and_audited(pack_dir: Path) -> None:
    audit = InMemoryAuditLog()
    pipeline, provider, tools, store = build(
        pack_dir, [calls("check_stock"), "I cannot check that."], policy=POLICY, audit=audit
    )

    await pipeline.answer(ask(), caller=MEMBER)

    assert tools.ran == []
    [result] = provider.requests[1].messages[-1].tool_results
    assert (result.content, result.is_error) == (DENIED, True)
    [entry] = audit.entries
    assert (entry.action, entry.user_id, entry.target) == (
        AuditAction.TOOL_DENIED, "m1", "check_stock",
    )  # fmt: skip
    [trace] = store.recent(1)
    assert "tool_denied" in [e.name for e in trace.events]
    assert trace.attributes["user"] == "m1"


async def test_tools_run_with_the_users_token_and_are_audited(pack_dir: Path) -> None:
    audit = InMemoryAuditLog()
    pipeline, _, tools, _ = build(
        pack_dir, [calls("find_order"), "Shipped."], policy=POLICY, audit=audit
    )

    await pipeline.answer(ask(), caller=MEMBER)

    assert tools.credentials == ["member-token"]
    [entry] = audit.entries
    assert (entry.action, entry.outcome, dict(entry.detail)) == (
        AuditAction.TOOL_CALL, "ok", {"arguments": {"id": 0}},
    )  # fmt: skip


async def test_only_the_user_who_started_a_run_can_decide_it(pack_dir: Path) -> None:
    audit = InMemoryAuditLog()
    pipeline, _, tools, _ = build(
        pack_dir, [calls("cancel_order"), "Cancelled."], policy=POLICY, audit=audit
    )
    paused = await pipeline.answer(ask(), caller=MEMBER)
    assert paused.run_id is not None
    decision = [Decision(paused.pending[0].id, approve=True)]

    with pytest.raises(RunNotFoundError):  # looks like no run at all to anyone else
        await pipeline.resume(paused.run_id, decision, OTHER)
    done = await pipeline.resume(paused.run_id, decision, MEMBER)

    assert done.completion.text == "Cancelled."
    assert tools.credentials == ["member-token"]  # the resuming user's own token
    assert [(e.action, e.outcome) for e in audit.entries] == [
        (AuditAction.APPROVAL, "approved"), (AuditAction.TOOL_CALL, "ok"),
    ]  # fmt: skip


async def test_retrieval_sees_only_the_labels_the_users_roles_allow(
    rag_pack_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAIN_PACK", str(rag_pack_dir))
    assert cli.main(["ingest"]) == 0
    (rag_pack_dir / "policy.yaml").write_text(
        "roles: {member: {read: []}, staff: {read: [internal]}}", encoding="utf-8"
    )
    config = yaml.safe_load((rag_pack_dir / "brain.yaml").read_text())
    config["access"] = {"jwt": {"secret_env": "BRAIN_JWT_SECRET"}}
    (rag_pack_dir / "brain.yaml").write_text(yaml.safe_dump(config))
    context = build_context(Settings(pack=rag_pack_dir), default_factory())

    staff = Caller(User("s1", frozenset({"staff"})))
    member = Caller(User("m1", frozenset({"member"})))
    for_staff = await context.pipeline.retrieve("anvil margin returns cost", staff)
    for_member = await context.pipeline.retrieve("anvil margin returns cost", member)

    assert "staff/margins.md" in {c.source for c in for_staff}
    assert "staff/margins.md" not in {c.source for c in for_member}
