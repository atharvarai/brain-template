"""The agent loop through the pipeline: steps, budgets, approvals, streaming, traces."""

from pathlib import Path

import pytest

from core.agent import DECLINED, Budget, Decision, DecisionError, RunClosedError, ToolActivity
from core.llm import StopReason, TextDelta, ToolChoice
from core.pipeline import Answer, AnswerStatus
from tests.unit.agent.helpers import FakeTools, ask, build, calls

pytestmark = pytest.mark.anyio


async def test_a_question_without_tools_is_one_step(pack_dir: Path) -> None:
    pipeline, provider, _, store = build(pack_dir, ["No tool needed."])

    answer = await pipeline.answer(ask())

    assert len(provider.requests) == 1
    assert provider.requests[0].tool_choice is ToolChoice.AUTO
    assert (answer.completion.text, answer.tools, answer.status) == (
        "No tool needed.", (), AnswerStatus.COMPLETE,
    )  # fmt: skip
    [trace] = store.recent(1)
    assert [s.name for s in trace.spans] == ["retrieve", "generate"]


async def test_steps_chain_until_the_model_answers(pack_dir: Path) -> None:
    """Multi-step: the second call depends on the first one's result."""
    pipeline, provider, tools, store = build(
        pack_dir, [calls("find_order"), calls("check_stock", prefix="d"), "It ships tomorrow."]
    )

    answer = await pipeline.answer(ask())

    assert [c.name for c in tools.ran] == ["find_order", "check_stock"]
    assert len(provider.requests) == 3
    third = provider.requests[2]
    assert third.tool_choice is ToolChoice.AUTO  # still within budget
    assert [m.role.value for m in third.messages] == [
        "user", "assistant", "user", "assistant", "user",
    ]  # fmt: skip
    assert answer.completion.text == "It ships tomorrow."
    assert [(a.name, a.step) for a in answer.tools] == [("find_order", 1), ("check_stock", 2)]
    assert answer.completion.usage.output_tokens == 1 + 1 + 3  # every step is counted
    [trace] = store.recent(1)
    steps = [(s.name, s.attributes.get("step")) for s in trace.spans[1:]]
    assert steps == [
        ("generate", 1), ("tool", 1), ("generate", 2), ("tool", 2), ("generate", 3),
    ]  # fmt: skip
    assert trace.attributes["steps"] == 3


async def test_a_failed_tool_is_reported_to_the_model(pack_dir: Path) -> None:
    pipeline, provider, _, _ = build(
        pack_dir, [calls("find_order"), "I could not look that up."], FakeTools(["find_order"])
    )

    answer = await pipeline.answer(ask())

    [result] = provider.requests[1].messages[-1].tool_results
    assert (result.content, result.is_error) == ("service down", True)
    assert answer.tools[0].ok is False


async def test_the_step_budget_forces_a_final_answer_without_tools(pack_dir: Path) -> None:
    pipeline, provider, _tools, store = build(
        pack_dir, [calls("find_order"), "Here is what I found."], budget=Budget(max_steps=2)
    )

    answer = await pipeline.answer(ask())

    last = provider.requests[-1]
    assert last.tool_choice is ToolChoice.NONE
    assert "step limit of 2 reached" in last.messages[-1].content
    assert answer.completion.text == "Here is what I found."
    [trace] = store.recent(1)
    assert [e.name for e in trace.events] == ["budget_reached"]


async def test_the_token_budget_counts_every_step(pack_dir: Path) -> None:
    pipeline, provider, _, _ = build(
        pack_dir, [calls("find_order"), "done"], budget=Budget(max_tokens=1000 - 999)
    )

    await pipeline.answer(ask())

    assert provider.requests[-1].tool_choice is ToolChoice.NONE
    assert "token budget of 1 used" in provider.requests[-1].messages[-1].content


async def test_calls_over_the_tool_budget_get_an_error_result(pack_dir: Path) -> None:
    pipeline, provider, tools, _ = build(
        pack_dir,
        [calls("find_order", "check_stock", "find_order"), "done"],
        budget=Budget(max_tool_calls=2),
    )

    await pipeline.answer(ask())

    assert len(tools.ran) == 2
    results = provider.requests[1].messages[-1].tool_results
    assert len(results) == 3  # every call the model made gets a result
    assert results[2].is_error and "limit of 2 tool calls" in results[2].content
    assert provider.requests[1].tool_choice is ToolChoice.NONE


# --- approval -------------------------------------------------------------------------------


async def test_a_call_that_changes_data_pauses_the_run(pack_dir: Path) -> None:
    pipeline, _provider, tools, store = build(pack_dir, [calls("cancel_order"), "Cancelled."])

    paused = await pipeline.answer(ask("cancel order 7"))

    assert paused.status is AnswerStatus.AWAITING_APPROVAL
    assert paused.run_id is not None
    [pending] = paused.pending
    assert (pending.name, dict(pending.arguments)) == ("cancel_order", {"id": 0})
    assert paused.completion.stop_reason is StopReason.TOOL_USE
    assert tools.ran == []  # nothing changed yet
    [trace] = store.recent(1)
    assert [e.name for e in trace.events] == ["approval_requested"]
    assert trace.attributes["run_id"] == paused.run_id

    done = await pipeline.resume(paused.run_id, [Decision(pending.id, approve=True)])

    assert [c.name for c in tools.ran] == ["cancel_order"]
    assert tools.ran[0].arguments == pending.arguments  # runs exactly as shown
    assert (done.status, done.completion.text) == (AnswerStatus.COMPLETE, "Cancelled.")
    assert done.completion.usage.output_tokens == 1 + 1  # both steps, across the pause
    resumed = store.recent(1)[0]
    assert resumed.operation == "resume"
    assert resumed.attributes["resumes"] == paused.run_id


async def test_an_approval_is_used_once(pack_dir: Path) -> None:
    pipeline, _, tools, _ = build(pack_dir, [calls("cancel_order"), "Cancelled."])
    paused = await pipeline.answer(ask())
    assert paused.run_id is not None
    decision = [Decision(paused.pending[0].id, approve=True)]

    await pipeline.resume(paused.run_id, decision)
    with pytest.raises(RunClosedError):
        await pipeline.resume(paused.run_id, decision)
    assert len(tools.ran) == 1


async def test_wrong_decisions_are_rejected_without_using_up_the_run(pack_dir: Path) -> None:
    pipeline, _, _tools, _ = build(pack_dir, [calls("cancel_order"), "Cancelled."])
    paused = await pipeline.answer(ask())
    assert paused.run_id is not None
    call_id = paused.pending[0].id

    with pytest.raises(DecisionError, match="no action waiting"):
        await pipeline.resume(paused.run_id, [Decision("other", approve=True)])
    with pytest.raises(DecisionError, match="no decision"):
        await pipeline.resume(paused.run_id, [])
    with pytest.raises(DecisionError, match="twice"):
        await pipeline.resume(paused.run_id, [Decision(call_id, True), Decision(call_id, False)])

    done = await pipeline.resume(paused.run_id, [Decision(call_id, approve=True)])
    assert done.status is AnswerStatus.COMPLETE


async def test_a_declined_action_is_not_run_and_the_model_is_told_why(pack_dir: Path) -> None:
    pipeline, provider, tools, _ = build(pack_dir, [calls("cancel_order"), "Okay, I left it."])
    paused = await pipeline.answer(ask())
    assert paused.run_id is not None

    done = await pipeline.resume(
        paused.run_id, [Decision(paused.pending[0].id, approve=False, note="changed my mind")]
    )

    assert tools.ran == []
    [result] = provider.requests[-1].messages[-1].tool_results
    assert result.is_error
    assert result.content == f"{DECLINED} Their note: changed my mind"
    assert done.completion.text == "Okay, I left it."


async def test_read_only_calls_run_at_once_and_results_keep_the_models_order(
    pack_dir: Path,
) -> None:
    pipeline, provider, tools, _ = build(pack_dir, [calls("cancel_order", "find_order"), "Done."])
    paused = await pipeline.answer(ask())
    assert [c.name for c in tools.ran] == ["find_order"]  # did not wait
    assert [c.name for c in paused.pending] == ["cancel_order"]
    assert paused.run_id is not None

    await pipeline.resume(paused.run_id, [Decision(paused.pending[0].id, approve=True)])

    results = provider.requests[-1].messages[-1].tool_results
    assert [r.name for r in results] == ["cancel_order", "find_order"]


async def test_auto_approved_tools_do_not_pause(pack_dir: Path) -> None:
    pipeline, _, tools, _ = build(
        pack_dir, [calls("cancel_order"), "Cancelled."], auto_approve=["cancel_order"]
    )

    answer = await pipeline.answer(ask())

    assert answer.status is AnswerStatus.COMPLETE
    assert [c.name for c in tools.ran] == ["cancel_order"]


# --- streaming ------------------------------------------------------------------------------


async def test_streaming_keeps_steps_apart_and_ends_with_the_answer(pack_dir: Path) -> None:
    pipeline, _, _, store = build(pack_dir, [calls("find_order"), "It ships tomorrow."])

    events = [event async for event in pipeline.stream(ask())]

    assert isinstance(events[0], ToolActivity)
    assert isinstance(events[-1], Answer)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text.strip() == "It ships tomorrow."
    [trace] = store.recent(1)
    assert "first_token" not in [e.name for e in trace.events]  # step 1 had no text


async def test_a_paused_stream_ends_with_an_answer_awaiting_approval(pack_dir: Path) -> None:
    pipeline, _, _, _ = build(pack_dir, [calls("cancel_order"), "Cancelled."])

    events = [event async for event in pipeline.stream(ask())]

    answer = events[-1]
    assert isinstance(answer, Answer)
    assert answer.status is AnswerStatus.AWAITING_APPROVAL
    assert answer.run_id is not None

    resumed = [e async for e in pipeline.resume_stream(answer.run_id, [
        Decision(answer.pending[0].id, approve=True)])]  # fmt: skip
    assert isinstance(resumed[0], ToolActivity)
    final = resumed[-1]
    assert isinstance(final, Answer) and final.completion.text == "Cancelled."
