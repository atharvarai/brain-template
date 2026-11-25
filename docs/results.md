# Measured results

Every number here came from running the brain, not from estimates. Three caveats apply to
all of them:

- **The model.** Answers came from Claude models through the Claude Code CLI on a
  subscription (`claude_code` profile: haiku, sonnet, opus). Costs are what the same tokens
  would cost at API list prices; the subscription is not billed per call.
- **The eval sets are small and not held out.** Skilvanta has 51 cases and Zipvaha 18. Some
  settings, such as the reranker, were chosen on the same cases they are scored on.
- **Latency includes the CLI.** Each model call starts a `claude -p` process, which adds
  about a second. An API adapter would be faster.

## Current state

| | Skilvanta (ed-tech) | Zipvaha (ride-hailing) |
|---|---|---|
| Eval cases | 51 | 18 |
| Answer evals passed | 50/51 (98%) | 18/18 |
| Retrieval evals passed (CI, no model) | 28/29 | 10/10 |
| Median latency per answer | 4.3 s | 4.0 s |
| Cost per answer at list price | $0.0034 | $0.0058 |
| Input tokens served from prompt cache | 94% | not measured separately |
| Unit and integration tests | 355 | (shared) |

The one Skilvanta failure is a Hinglish question. The English-only reranker drops the
section that holds the answer; a multilingual reranker fixes it at about 1.5 s per query.

## What each measurement changed

| Milestone | Measured | Decision |
|---|---|---|
| M4 retrieval | Section-level retrieval 20/23 with hybrid search alone, 22/23 with a tiny cross-encoder reranker (76 ms per query), 23/23 with a multilingual one (1.5 s, 570 MB) | Tiny reranker on for Skilvanta; multilingual documented as the fix for Hinglish |
| M4 evals | First full run 24/27. A placement question failed because filler words outranked the real match in keyword search | Stopwords in keyword search; section-level `retrieves` checks, because document-level ones hid the miss |
| M5 tools | Claude Code tool emulation through `--json-schema` nearly doubled input tokens per question (1,776 to 3,292) | Kept for M5, revisited in M6 |
| M6 agent | Under JSON-mode emulation, one refund case passed 0 of 6 runs; with tools off it passed 6 of 6. Answers were thinner inside a JSON string | Replaced with a `<function_calls>` text protocol: streaming back, input tokens down 30% on the full run ($0.51 to $0.38) |
| M6 approvals | After a declined booking, the model described the decline as a failure (passed 2 of 3 runs) | Clearer result text plus a system prompt rule (v4): 3 of 3 |
| M7 access | With write tools available, policy answers started offering actions and hedging on amounts. Refund cases passed 12 of 18 runs | System prompt v5: concrete figures, no unrequested offers. 14 of 18, full set back to 45/46 |
| M8 guardrails | Injection cases 6/6. Blocked messages took 0.0 s (no model call). The numbers grounding check flagged 4 of 49 answers: two correct derived figures, two a timestamp parsing bug | Parsing fixed. The check stays flag-only, because derived figures are legitimate |
| M9 prompt cache | 94% of input tokens read from the prompt cache. The same traffic costs $0.448 without cache pricing and $0.168 with it | Cache prices in `brain.yaml`; Anthropic breakpoints on tools, system and agent steps |
| M9 semantic cache | On 20 question pairs, a near miss with the opposite answer scored 0.974, above every true paraphrase. The only threshold with no wrong answers caught nothing | Semantic cache ships off by default; `make cache-check` measures it per pack |
| M9 router | Fixed mid: 50/51, $0.0034 per answer. Heuristic: 48/51, $0.0029, and the small model refused an admin a permitted document. Model router: 50/51, $0.0033, +3 s | Skilvanta stays on fixed mid: prompt caching already took most of the cost |
| M10 second pack | Zipvaha needed one line in `core/` (a validation bug), passed 15/18 at first and 18/18 after two pack-side fixes | [agnosticism.md](agnosticism.md) |

## Mistakes caught along the way

These are worth keeping, because they show what the evals and the guards are for.

- **The agnosticism test caught company words in `core/` twice** ("course", "Skilvanta",
  "mentor" in comments).
- **A price change during a background comparison confounded the M9 cost numbers.** It was
  caught because a 64% saving from 4 rerouted questions could not be right. All runs were
  re-priced with one price book.
- **A milestone's edit failed silently and left an eval-validation gap.** The second pack
  found it.
