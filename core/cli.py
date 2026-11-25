"""Operator commands, run as ``python -m core.cli <command>``.

- ``ingest``  build or update the document index
- ``search``  show what retrieval returns for a query
- ``eval``    run the pack's eval cases and fail below the pass-rate threshold
- ``traces``  list recent traces with a summary, or show one trace in full
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Callable

from core.api.app import build_context
from core.api.context import AppContext
from core.config import PackError, Settings, load_pack
from core.evals import (
    JUDGE_PROMPT,
    AnswerTarget,
    EvalError,
    EvalRunner,
    EvalTarget,
    LLMJudge,
    ReportArchive,
    RetrievalTarget,
    compare,
    load_cases,
    render,
)
from core.llm import default_factory
from core.observability import PriceBook, summarize, trace_to_dict
from core.prompts import PromptError
from core.rag import RagError, build_rag

Command = Callable[[argparse.Namespace, Settings], int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m core.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="build or update the document index")
    ingest.add_argument("--rebuild", action="store_true", help="drop the index first")

    search = commands.add_parser("search", help="show what retrieval returns for a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=None, help="number of results")
    search.add_argument(
        "--all-labels", action="store_true", help="include restricted documents (debugging)"
    )

    evaluate = commands.add_parser("eval", help="run the pack's eval cases")
    evaluate.add_argument(
        "--retrieval", action="store_true", help="retrieval checks only: no model, runs in CI"
    )
    evaluate.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    evaluate.add_argument("--case", action="append", default=[], help="only this case id")
    evaluate.add_argument("--min-pass-rate", type=float, default=None, help="override the bar")
    evaluate.add_argument("--no-save", action="store_true", help="do not archive the report")

    traces = commands.add_parser("traces", help="recent traces and a summary")
    traces.add_argument("-n", type=int, default=20, help="how many traces")
    traces.add_argument("--operation", default=None, help="only chat, title, eval, judge...")
    traces.add_argument("--id", default=None, help="print one trace in full, as JSON")
    return parser


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    pack = load_pack(settings.pack)
    rag = build_rag(pack, settings.data_dir)
    if rag is None:
        print(f"{pack.root}: brain.yaml has no 'rag' section, nothing to do")
        return 1
    report = rag.ingestor.run(rebuild=args.rebuild)
    print(
        f"{pack.config.name}: {report.added} added, {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.removed} removed. "
        f"Index now holds {report.chunks} chunks."
    )
    return 0


def _search(args: argparse.Namespace, settings: Settings) -> int:
    pack = load_pack(settings.pack)
    rag = build_rag(pack, settings.data_dir)
    if rag is None:
        print(f"{pack.root}: brain.yaml has no 'rag' section, nothing to do")
        return 1
    allowed = None if args.all_labels else frozenset[str]()
    hits = asyncio.run(rag.retriever.search(args.query, args.k or rag.top_k, allowed))
    if not hits:
        print("no results (is the index built? run ingest)")
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        labels = f" labels={sorted(chunk.labels)}" if chunk.labels else ""
        print(f"{rank}. {hit.score:.4f}  {chunk.source} :: {chunk.heading}{labels}")
        print(f"   {chunk.text[:160].replace(chr(10), ' ')}")
    return 0


def _eval_target(context: AppContext, retrieval_only: bool) -> tuple[EvalTarget, LLMJudge | None]:
    if retrieval_only:
        return RetrievalTarget(context.pipeline), None
    config = context.pack.config
    judge = LLMJudge(
        context.gateway.for_tier(config.evals.judge_tier),
        context.prompts.get(JUDGE_PROMPT),
        context.tracer,
    )
    return AnswerTarget(context.pipeline, PriceBook(config.models.prices)), judge


def _eval(args: argparse.Namespace, settings: Settings) -> int:
    context = build_context(settings, default_factory())
    config = context.pack.config
    cases = load_cases(context.pack.eval_cases)
    if args.tag:
        cases = [case for case in cases if set(args.tag) & set(case.tags)]
    if args.case:
        cases = [case for case in cases if case.id in args.case]
    if not cases:
        print("no cases match the filters")
        return 1

    target, judge = _eval_target(context, args.retrieval)
    runner = EvalRunner(target, judge, config.name, context.model_profile, config.evals.concurrency)
    report = asyncio.run(runner.run(cases))

    archive = ReportArchive(context.pack.state_dir(settings.data_dir) / "evals")
    previous = archive.latest(report.mode.value)
    threshold = args.min_pass_rate if args.min_pass_rate is not None else config.evals.min_pass_rate
    print(render(report, threshold, compare(report, previous) if previous else None))
    if not args.no_save:
        print(f"Report saved to {archive.save(report)}")
    return 0 if report.results and report.pass_rate >= threshold else 1


def _traces(args: argparse.Namespace, settings: Settings) -> int:
    context = build_context(settings, default_factory())
    if args.id:
        trace = context.traces.get(args.id)
        if trace is None:
            print(f"no trace with id {args.id}")
            return 1
        print(json.dumps(trace_to_dict(trace), indent=2, ensure_ascii=False))
        return 0

    traces = context.traces.recent(args.n, args.operation)
    if not traces:
        print("no traces yet: send a request to the brain first")
        return 0
    feedback = context.traces.feedback_for([t.trace_id for t in traces])
    for t in traces:
        cost = f"${t.cost_usd:.5f}" if t.cost_usd is not None else "-"
        rating = feedback[t.trace_id].rating.value if t.trace_id in feedback else ""
        print(
            f"{t.started_at.astimezone():%Y-%m-%d %H:%M:%S}  {t.trace_id[:12]}  {t.operation:<6} "
            f"{t.status.value:<5} {t.duration_ms:>6}ms  {t.model or '-':<10} "
            f"{t.input_tokens:>6}/{t.output_tokens:<5} {cost:>9}  "
            f"cited={len(t.cited)}/{len(t.retrieved)} {rating}".rstrip()
        )
        if t.error:
            print(f"    error: {t.error}")
    s = summarize(traces, feedback)
    total_cost = f"${s.cost_usd:.4f}" if s.cost_usd is not None else "unknown"
    print(
        f"\n{s.count} traces, {s.errors} errors ({s.error_rate:.0%}); "
        f"p50 {s.p50_ms}ms, p95 {s.p95_ms}ms; tokens {s.input_tokens} in / "
        f"{s.output_tokens} out; cost {total_cost}; feedback +{s.thumbs_up} / -{s.thumbs_down}"
    )
    return 0


COMMANDS: dict[str, Command] = {
    "ingest": _ingest,
    "search": _search,
    "eval": _eval,
    "traces": _traces,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, Settings())
    except (PackError, RagError, EvalError, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
