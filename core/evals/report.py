"""Eval reports: saved as JSON per run, printed as text, compared with the previous run."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.evals.runner import EvalReport
from core.evals.targets import EvalMode


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "pack": report.pack,
        "mode": report.mode.value,
        "profile": report.profile,
        "started_at": report.started_at.isoformat(),
        "duration_ms": report.duration_ms,
        "pass_rate": report.pass_rate,
        "passed": report.passed,
        "total": len(report.results),
        "skipped": list(report.skipped),
        "cases": [
            {
                "id": r.case_id,
                "passed": r.passed,
                "error": r.error,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                ],
                "trace_id": r.outcome.trace_id if r.outcome else None,
                "latency_ms": r.outcome.latency_ms if r.outcome else None,
                "answer": r.outcome.answer if r.outcome else None,
                "retrieved": list(r.outcome.retrieved) if r.outcome else [],
                "tools_called": list(r.outcome.tools_called) if r.outcome else [],
                "approvals_requested": list(r.outcome.approvals_requested) if r.outcome else [],
                "blocked": r.outcome.blocked if r.outcome else False,
                "tier": r.outcome.tier if r.outcome else None,
                "guard_flags": list(r.outcome.guard_flags) if r.outcome else [],
            }
            for r in report.results
        ],
    }


class ReportArchive:
    """Keeps every run's report so the next run can show what got better or worse."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def latest(self, mode: str) -> dict[str, Any] | None:
        runs = sorted(self._directory.glob(f"{mode}-*.json"))
        return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None

    def save(self, report: EvalReport) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        stamp = report.started_at.strftime("%Y%m%dT%H%M%S")
        path = self._directory / f"{report.mode.value}-{stamp}.json"
        path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class Comparison:
    """This run against the previous one, over the cases both ran (filters can differ)."""

    previous_at: str
    shared: int  # cases in both runs
    previous_rate: float  # pass rate on the shared cases, then and now
    current_rate: float
    regressions: tuple[str, ...]  # passed last time, fail now
    fixes: tuple[str, ...]  # failed last time, pass now


def compare(report: EvalReport, previous: dict[str, Any]) -> Comparison:
    before = {case["id"]: case["passed"] for case in previous["cases"]}
    now = {result.case_id: result.passed for result in report.results}
    shared = before.keys() & now.keys()

    def rate(results: dict[str, bool]) -> float:
        return sum(results[c] for c in shared) / len(shared) if shared else 0.0

    return Comparison(
        previous_at=previous["started_at"][:19].replace("T", " "),
        shared=len(shared),
        previous_rate=rate(before),
        current_rate=rate(now),
        regressions=tuple(sorted(c for c in shared if before[c] and not now[c])),
        fixes=tuple(sorted(c for c in shared if not before[c] and now[c])),
    )


def _ratio(passed: int, total: int) -> str:
    return f"{passed}/{total}"


def render(report: EvalReport, threshold: float, comparison: Comparison | None = None) -> str:
    lines = [
        f"{report.pack} evals: {report.mode.value} mode, profile {report.profile}, "
        f"{len(report.results)} cases"
    ]
    width = max((len(r.case_id) for r in report.results), default=10)
    for result in report.results:
        mark = "PASS" if result.passed else "FAIL"
        seconds = f"{result.outcome.latency_ms / 1000:5.1f}s" if result.outcome else "    -"
        problem = result.error or "; ".join(
            f"{c.name}: {c.detail}" for c in result.checks if not c.passed
        )
        lines.append(f"  {mark}  {result.case_id:<{width}}  {seconds}  {problem}".rstrip())
    if report.skipped:
        lines.append(f"  skipped (no {report.mode.value} checks): {len(report.skipped)} cases")

    verdict = "OK" if report.pass_rate >= threshold else "BELOW THRESHOLD"
    lines += [
        "",
        f"Pass rate {_ratio(report.passed, len(report.results))} = {report.pass_rate:.1%} "
        f"(threshold {threshold:.0%}): {verdict}",
        "By check: "
        + ", ".join(f"{name} {_ratio(*counts)}" for name, counts in report.by_check().items()),
    ]
    if tags := report.by_tag():
        lines.append("By tag:   " + ", ".join(f"{t} {_ratio(*c)}" for t, c in tags.items()))
    tiers = Counter(r.outcome.tier for r in report.results if r.outcome and r.outcome.tier)
    if tiers:
        lines.append("Tiers:    " + ", ".join(f"{t} {n}" for t, n in sorted(tiers.items())))
    p50, p95 = report.latency_ms(0.5) / 1000, report.latency_ms(0.95) / 1000
    usage = ""
    if report.mode is EvalMode.ANSWER:
        tokens_in, tokens_out = report.tokens
        cost = report.cost_usd
        per = f", ${cost / len(report.results):.4f} per case" if cost and report.results else ""
        cost_text = (
            f"${cost:.4f} at list price{per}" if cost is not None else "cost unknown (no price)"
        )
        usage = f" tokens {tokens_in} in / {tokens_out} out; {cost_text};"
    lines.append(
        f"Latency p50 {p50:.1f}s, p95 {p95:.1f}s;{usage} wall time {report.duration_ms / 1000:.1f}s"
    )
    if comparison and comparison.shared:
        delta = (comparison.current_rate - comparison.previous_rate) * 100
        lines.append(
            f"Previous run ({comparison.previous_at}), on the {comparison.shared} cases both ran: "
            f"{comparison.previous_rate:.1%} then, {comparison.current_rate:.1%} now "
            f"({delta:+.1f} points)"
        )
        if comparison.regressions:
            lines.append(f"  regressions: {', '.join(comparison.regressions)}")
        if comparison.fixes:
            lines.append(f"  fixed: {', '.join(comparison.fixes)}")
    return "\n".join(lines)
