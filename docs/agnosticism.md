# Agnosticism log

The template claims that a company is a folder: everything specific to it lives in a
domain pack, and `core/` never changes. This file records the test of that claim.

## Method (M10)

Skilvanta (ed-tech) was built alongside the core in M0 to M9. Zipvaha, a fictional
ride-hailing company's driver support, was then built under one rule: change only
`examples/rides/`. The commit before the work was recorded, and `git diff` against it was
checked at the end. Any change needed under `core/` counts as a finding.

Zipvaha shares nothing with Skilvanta's domain:

| | Skilvanta | Zipvaha |
|---|---|---|
| Users | learners, mentors, support agents, admins | drivers, operations |
| Documents | 19 (courses, refunds, mentorship...) | 8 (payouts, incentives, appeals...) |
| Restricted labels | `internal`, `admin_only` | `internal` |
| Tools | enrollments, progress, mentor slots, booking, refunds, mentee progress | payouts, bonus progress, payout disputes, driver lookup |
| Language | English, some Hinglish | English and Hinglish |

## Result

**One line changed under `core/`, and it was a bug, not a bias.**

| # | Finding | Kind | What changed |
|---|---|---|---|
| 1 | An eval case whose only expectation is `"blocked": true` was rejected as having no expectation. M8 added `blocked` but not to the "at least one expectation" check, and every Skilvanta case that used it also had another check | Core bug, hidden by the first pack's cases | `core/evals/cases.py`, one line, plus a test |

Nothing else in `core/` changed. Everything Zipvaha needed came from its folder:

- roles, labels, tool permissions
- persona, documents, MCP server, eval cases
- prices, guardrail settings, cache and router choices

## Lessons from building the second pack

These are about packs, not the core, so they belong in the "make your own brain" guide.

- **A persona must not fight the core rules.** "Many drivers write in Hinglish: reply in
  the same mix" made the model answer English questions in Hinglish, against the core rule
  to answer in the user's language. Three cases failed. `persona.v2` says to reply in the
  driver's own language.
- **`excludes` also matches negations.** A case banned "3400" (the wrong bonus total), and
  the correct answer said "it will not total Rs 3,400". Use a rubric when the point is what
  the answer claims, and `excludes` for text that must never appear at all, such as leaked
  figures.
- **Tool descriptions decide whether tools get used.** This was learned on Skilvanta in M8.
  Zipvaha's tools say exactly what they return, and every tool case passed on the first run.
- **Each demo MCP server repeats about 30 lines of token checking.** A real backend has its
  own auth, so this is not core work, but a shared example helper would shorten new packs.

## Numbers

| Pack | Eval cases | Pass (answer mode, Claude Code) | Retrieval mode (CI) |
|---|---|---|---|
| Skilvanta | 51 | 50/51 | 28/29 |
| Zipvaha | 18 | 18/18 (15/18 before the two pack fixes above) | 10/10 |
