# Brain Template

An industry-agnostic template for a company's AI microservice: its "brain".

Clone it, add one folder that describes your company (docs, tools, roles, prompts, eval
cases), and you get an assistant built from clearly separated AI layers: model, prompts,
RAG, tools (MCP), agent, access control, guardrails, cache and router, observability and evals.

**Status: early.** Milestones M0 to M9 are done. The service answers from a company's
documents with citations, and from the company's own systems through MCP tools. An agent
loop takes several steps within budgets, and actions that change data wait for the user's
approval.

Every request carries the user's token. Their roles decide which documents they can read
and which tools they can call, and tools run as them. Guardrails sit around every model
call.

Prompt caching is measured from token usage. Repeated questions are answered from a cache
scoped to what the user may read. A router can send easy questions to a smaller model, and
the evals decide whether it should.

Underneath are a vendor-neutral model layer with failover, versioned prompts, and
schema-validated structured output. Every request is traced, actions are audited, and each
pack carries an eval set that gates changes. See [docs/PLAN.md](docs/PLAN.md) for the
design and the milestone list.

## How it is organised

| Path | What it holds |
|---|---|
| `core/` | The generic brain. One folder per layer. Never imports from `examples/`. |
| `examples/edtech/` | Domain pack for Skilvanta, a **fictional** ed-tech company (synthetic data). |
| `tests/boundary/` | Tests that fail if company-specific words or imports leak into `core/`. |
| `docs/` | Plan and design notes. |

A company is a **domain pack**: a folder with a `brain.yaml` and its own prompts, docs,
policies and eval cases. The process picks a pack with the `BRAIN_PACK` environment variable.

## Model-agnostic by design

Code asks for a tier (`small`, `mid`, `large`), never for a vendor or a model name.
`brain.yaml` maps each tier to an ordered fallback chain of `{provider, model}` entries.
Switching vendor is a config change:

```yaml
models:
  active: anthropic            # or override per process: BRAIN_MODEL_PROFILE=ollama
  profiles:
    anthropic:
      small: [{provider: anthropic, model: claude-haiku-5-5}]
      mid:                     # first entry is tried first, the next is the fallback
        - {provider: anthropic, model: claude-sonnet-5-5}
        - {provider: ollama, model: "llama3.1:8b"}
      large: [{provider: anthropic, model: claude-opus-5-5}]
```

| Provider (`Provider` enum) | Needs | Notes |
|---|---|---|
| `claude_code` | Claude Code CLI, logged in | No API key. Development use: one prompt per call. |
| `ollama` | Ollama running locally | Free and offline. Host from `OLLAMA_HOST`. |
| `anthropic` | `uv sync --extra anthropic`, `ANTHROPIC_API_KEY` | |
| `openai` | `uv sync --extra openai`, `OPENAI_API_KEY` | |
| `fake` | nothing | Deterministic echo, for tests. |

Each provider is one adapter in `core/llm/adapters/`. Adding a vendor means writing one
adapter and registering one builder in `core/llm/factory.py`. A chain fails over on outages,
timeouts and rate limits, never on a rejected request, and a circuit breaker stops calls to
a provider that keeps failing.

The Claude Code, fake and failover paths are tested against the real thing. The Anthropic,
OpenAI and Ollama adapters are tested with mocked HTTP only; they have not been run against
the live services.

## Prompts

Prompts are files, versioned by filename: `<name>.v<version>.md`.

- `core/prompts/builtin/` holds generic skeletons. `system.v1.md` carries the rules every
  brain follows and has slots for the company name, description and persona.
- A pack's `prompts/` folder holds the company's own text. `persona.v1.md` is required. A
  pack can also override any built-in prompt by adding a file with the same name.
- The newest version is used unless `brain.yaml` pins one (`prompts: {pinned: {persona: 1}}`),
  so rolling back a bad prompt change is a one-line edit.
- Every answer reports the name, version and checksum of each prompt behind it. The checksum
  changes if someone edits a file without bumping its version.

Templates only support `{{name}}` placeholders. Rendering fails on a missing or an unused
variable, and inserted text is never rendered again. Text the service does not control (user
input, and later documents and tool results) is wrapped in XML-style tags that it cannot
close early, and the prompts tell the model to treat tagged content as data.

## Structured output

`complete_structured(provider, request, SomePydanticModel)` returns a validated object from
any provider. It sends the schema through the vendor's native feature, validates the reply,
and retries once with the validation errors if needed. `POST /v1/title` is the first task
built on it: it names a conversation, on the small tier.

## Retrieval (RAG)

A pack's `docs/` folder holds Markdown files. Optional front matter sets a `title` and
`labels`:

```markdown
---
title: Refund Policy
labels: []          # [] = public. [internal] or [admin_only] = restricted
---
```

`make ingest` builds the index and `make search q="..."` shows what retrieval returns.

- **Ingestion is incremental.** Unchanged files are skipped, edited files are re-embedded,
  and deleted files are removed from the index, so stale text is never retrieved.
- **Chunking follows the document.** Files are split on headings, and each chunk keeps its
  title and heading path.
- **Search is hybrid.** Semantic search (local embedding model, no API key) and keyword
  search (BM25 with stemming, stopwords dropped) are merged with reciprocal rank fusion.
- **Optional reranking.** A local cross-encoder can re-order the top candidates
  (`rag.reranker` in `brain.yaml`). It is off by default; turn it on when your evals say so.
- **Answers cite their sources.** Retrieved chunks are numbered, the model cites them as
  `[1]`, and the response lists every source with a `cited` flag.
- **Restricted documents stay hidden for now.** Labels are stored with every chunk, and until
  the access layer (M7) exists only unlabelled documents are ever retrieved.

Everything is behind interfaces (`Loader`, `Chunker`, `Embedder`, `ChunkStore`, `Retriever`).
The shipped store is a zero-install SQLite file with brute-force vector search, which suits
development and small document sets. A production vector database adapter is not written yet.
The default embedding and reranking models are English-only; set multilingual models in
`brain.yaml` for other languages (for example `BAAI/bge-reranker-v2-m3`, which is much slower).

## Tools (MCP)

Tools come only from MCP servers listed in `brain.yaml`. The brain discovers their names,
descriptions and argument schemas at start-up, so `core/` never names a tool and the same
code works for any company's server.

```yaml
tools:
  servers:
    skilvanta: {transport: stdio, command: [python, mcp_server/server.py]}  # local dev
    # skilvanta: {transport: http, url: "http://backend.internal:8792/mcp"}  # production
  timeout_seconds: 10
  max_result_chars: 4000
```

For each question the model sees the documents and the tools. What happens when it asks
for tools is the agent's job (next section).

Failures never end the request. Each one goes back to the model as an error result, and the
model tells the user it could not get the data:

- a tool reports an error (for example "not enrolled in that course")
- arguments fail the tool's schema, or the model names a tool that does not exist
- a call takes longer than `timeout_seconds`
- the server cannot be reached. Internal details stay in the log, and the brain reconnects
  in the background with growing pauses, so a restarted server is picked up again.

A server that is down at start-up is skipped. `/healthz` then reports `degraded` and names
it, and the brain answers from documents alone. Results longer than `max_result_chars` are
cut, with a note saying so. Streamed chats send a `tool` event for each tool run, and the
final `done` event lists them.

Skilvanta ships a demo MCP server (`examples/edtech/mcp_server/server.py`) with synthetic
learner data. It has three read-only tools (enrollments, course progress, mentor slots) and
two that change data and so need approval (book a mentor session, request a refund).

```bash
make tools                                                # list discovered tools
make tools tool=get_course_progress args='{"course": "DSA"}'  # run one
make mcp-server                                           # serve it over HTTP on :8792
```

Each adapter maps tool calling onto its vendor's native feature: Anthropic `tool_use`
blocks, OpenAI function calls, Ollama `tool_calls`. The Claude Code CLI runs tools in its
own loop, which would take control away from the brain, so that adapter emulates tool
calling. It lists the tools in the system prompt. The model asks for calls by ending its
reply with a `<function_calls>` block, which the adapter parses and keeps out of the
streamed text.

A first version constrained the reply to JSON with `--json-schema` instead. The evals
caught three costs:

- Answers got thinner: one refund case passed 0 of 6 runs, against 6 of 6 without tools.
- Input tokens per question nearly doubled.
- Streaming was impossible.

The text protocol replaced it.

## Agent

A question that needs tools runs as a loop. One model step either answers or asks for
tools; the brain runs them and feeds the results back, and the loop repeats. Budgets stop
runaway loops. When one runs out, the model gets one last step with tools switched off, so
every run ends with an answer:

```yaml
agent:
  max_steps: 6            # model calls per question
  max_tokens: 50000       # input + output, summed over the steps
  max_tool_calls: 8
  auto_approve: []        # tools that may change data without asking
  approval_ttl_minutes: 60
```

**Approval.** MCP servers mark tools that change nothing with `readOnlyHint`. Those run at
once. Any other tool pauses the run, unless it is listed in `auto_approve`, so a server
that forgets the hint fails safe.

A paused answer has `status: "awaiting_approval"`, a `run_id`, and the pending actions with
their exact arguments. A streamed chat also sends an `approval` event. The client shows them
to the user and posts the decisions:

```bash
curl http://127.0.0.1:8000/v1/runs/<run_id>/decisions -H 'content-type: application/json' \
  -d '{"decisions": [{"call_id": "<call_id>", "approve": true}]}'
```

The run then carries on to its answer, streamed or as JSON, like `/v1/chat`. The rules:

- Approved actions run exactly as shown.
- Declined ones go back to the model as "not run, the user declined".
- A run can be decided only once (a replay gets 409), and it expires after
  `approval_ttl_minutes` (404).
- Wrong or missing call ids get 422 and do not use up the run.

Paused runs are stored in SQLite, so they survive a restart. Each step is traced: generate
and tool spans carry their step number, and the trace records events such as
`approval_requested`, `approval_decided` and `budget_reached`. A resumed run gets its own
trace, linked to the paused one.

## Access

With an `access` section in `brain.yaml`, every request needs a bearer token (a JWT from the
company's backend). A missing or bad token gets a 401, and the refusal is audited.

```yaml
access:
  jwt: {algorithm: HS256, secret_env: BRAIN_JWT_SECRET, issuer: skilvanta-backend,
        audience: skilvanta-brain}        # or RS256 with public_key_file
  policy: policy.yaml
  dev_users: seed/users.yaml              # demo users for local tokens and evals
```

The pack's `policy.yaml` maps roles to readable labels and callable tools:

```yaml
roles:
  learner: {read: [],                      tools: [get_my_enrollments, book_mentor_session, ...]}
  mentor:  {read: [internal],              tools: [get_mentee_progress]}
  admin:   {read: [internal, admin_only], tools: ["*"]}
```

Anything not granted is denied. Unlabelled documents are open to everyone. The rules are
applied at three points:

- **Retrieval:** the store returns only chunks whose every label the user may read. The
  filter is part of the SQL query, so restricted text never leaves the store.
- **Tools:** the model is shown only the tools the user's roles allow. Each call is checked
  again before it runs, and a refused call is audited.
- **The tool server:** tools run as the user. The token goes to the MCP server in the
  `Authorization` header over HTTP, or in the request `_meta` over stdio (development
  only). The server checks it and applies its own rules. In Skilvanta's demo, for example,
  a mentor sees only their own mentees, even though the brain's policy lets mentors call
  the tool.

A paused run can be decided only by the user who started it; to anyone else it looks like
no run at all. Feedback on an answer is likewise accepted only from the user who got it.

The audit log (`make audit`) records every tool call, every refused call, every approval
decision and every rejected token, each with the user and their roles. It is kept apart
from traces.

Without an `access` section the brain is open: every request is anonymous, sees unlabelled
documents only, and may call every tool. A warning is logged at start-up.

```bash
make token as=riya       # sign a token for a demo user (dev secret from the Makefile)
make tools tool=get_course_progress args='{"course": "DSA"}' as=riya
make audit
```

## Guardrails

Checks run around every model call. Each is a small class behind one `Guardrails` object,
configured per pack:

```yaml
guardrails:
  pii: {kinds: [email, phone, card, ip_address, in_pan, in_aadhaar]}
  injection: {detector: heuristic, action: block, scan_sources: true}
  grounding: {check: numbers, action: flag}
  prompt_leak: {action: block}
```

| Check | Where | What happens |
|---|---|---|
| Prompt injection | the user's message | Refused with a fixed reply and no model call, or flagged. `heuristic` matches known phrasings for free; `model` asks the small tier and catches other languages |
| Injected sources | retrieved documents, tool results | A document that gives orders is dropped. A tool result is kept but marked as data |
| Prompt leak | the answer | An answer that copies a run of the system prompt's rules is replaced |
| Grounding | the answer | Figures that appear in no document, tool result or question are flagged |
| PII | traces, logs, audit log | Emails, phone numbers, cards (Luhn), IP addresses, PAN, Aadhaar (Verhoeff) become `[EMAIL]` and so on |

Every finding lands on the trace and on the answer (`guard_flags`); blocks also go to the
audit log. A streamed answer is checked when it ends. If a guard replaces it, a `replace`
event carries the text the client must show instead. A refused message comes back with
`status: "blocked"` and provider `none`.

What the Skilvanta evals showed:

- **Patterns plus the model's own rules.** A Hinglish injection gets past the English
  patterns, but the model refuses it anyway. Adding the `model` detector is a config change.
- **The grounding check flags derived figures.** It fired on 4 of 49 answers in one run.
  Two were correct arithmetic: a fee after a 30% scholarship, and attendance as a
  percentage. Two were a parsing bug, now fixed. That is why it flags rather than blocks.
- **Redaction errs towards caution.** The company's own support address is redacted in
  stored traces too.

The Skilvanta pack carries two planted injections on purpose: a learner tip page with an
injected "SYSTEM:" line, and a mentor note in the demo backend data that offers a fake
refund code. The evals check that neither has any effect.

## Cost: prompt caching, answer cache, router

**Prompt caching.** The system prompt is identical on every request (documents go in the
user turn), so vendors can cache it. Usage records cached tokens separately
(`cache_read_tokens`, `cache_write_tokens`), and prices in `brain.yaml` can give them their
own rates. Without cache rates, cached tokens are costed at the input rate, so a missing
price never makes a run look cheaper than it is. The Anthropic adapter marks breakpoints
explicitly: tools and the system prompt always, the conversation only on agent steps that
will be followed by another. `make traces` shows the share of input served from cache.

**Answer cache.** An exact repeat of a question is answered without a model call. Entries
are shared only within a scope. The scope is a hash of the user's readable labels, the
prompt versions, the model profile and the document index version, so an entry can't cross
an access boundary and goes stale on any prompt or document change. Only answers that are
the same for everyone in the scope are stored: no tools ran, no guardrail noticed anything,
and the model finished normally. Evals bypass the cache unless run with `--cache`.

A semantic cache (reusing the answer to a question that means the same) is built but off by
default. `make cache-check` scores the pack's own question pairs at every threshold. On
Skilvanta's pairs the embedder scored near misses above true paraphrases: "What happens if
I miss a mentor session?" and "...if my mentor misses a session?" scored 0.974, and their
answers are opposite. No threshold was both useful and safe. Questions that differ in any
figure ("refund on day 5" and "day 25") are never matched, whatever the score.

**Router.** `fixed` uses one tier. `heuristic` sends short, impersonal lookups to the small
tier using word rules. `model` lets the small tier classify each question. Each run below
covers all 51 Skilvanta evals through Claude Code, with every run priced at the same rates:

| Router | Pass rate | Answers on the small tier | Cost per answer | Median latency |
|---|---|---|---|---|
| fixed (mid) | 50/51 | 0 | $0.0034 | 4.3 s |
| heuristic | 48/51 | 11 | $0.0029 | 4.2 s |
| model | 50/51 | 4 | $0.0033 | 7.6 s |

Prompt caching already did most of the work: 94% of input tokens came from the cache,
cutting the cost of the same traffic by about 62%. The heuristic router saved 14% more but
lost a case for real: the small model refused an admin a document the admin may read. The
model router's own calls ate its savings and added latency. Skilvanta therefore stays on
`fixed`. A pack with costlier traffic, or a small model that holds up on its evals, can
switch with one line, or try one per process with `BRAIN_ROUTER=heuristic make eval`.

## Observability

Every request (chat, title, eval, judge) writes one trace:

- timings per step (retrieve, generate) and time to first token when streaming
- provider and model that answered, tokens, and cost from the prices in `brain.yaml`
- prompt names, versions and checksums, plus the chunks retrieved and the ones cited
- failover events, such as a provider failing before the next one answered
- the question and answer text, only if the pack sets `observability.capture_content`

Traces go to a SQLite file in `.brain/` and to one JSON log line each. Sinks are behind a
`TraceSink` interface, so a company can add its own backend. Responses carry a `trace_id`;
`POST /v1/feedback` records a thumbs up or down against it.

```bash
make traces                                    # recent traces, p50/p95, cost, feedback
uv run python -m core.cli traces --id <id>     # one trace in full
```

## Evals

A pack's `evals/cases.jsonl` lists questions and what a good answer shows:

```json
{"id": "placement-eligibility", "question": "What do I need to qualify for placement support?",
 "tags": ["placement"],
 "expect": {"contains_all": ["80%", "75%"], "retrieves": ["placement-support.md#Eligibility"],
            "rubric": "Lists all three requirements: 80% assignments, 75% attendance, 2 mock interviews."}}
```

Each expectation becomes a check: `contains_all`, `contains_any`, `excludes` (for leaks),
`retrieves` and `cites` (a document, or one section with `doc.md#Heading`), and `rubric`,
graded by a judge model.

```bash
make eval-retrieval   # retrieval checks only: no model, about 2 s, runs in CI
make eval             # asks the model every case and grades it
```

Both print per-case results, pass rate by check and tag, latency and cost. They also
compare with the previous run, listing regressions and fixes, and exit non-zero below
`evals.min_pass_rate`.

A `calls` expectation checks tool use: `{"calls": ["get_my_enrollments"]}` needs that tool
to have run, and `{"calls": []}` means no tool may run. `asks_approval` works the same way
for actions that paused for approval. A case's `"approve": true` or `false` is the decision
the eval gives when the brain asks.

Each case runs as a demo user with a real signed token (`evals.default_user`, or the case's
`"user"`), so retrieval filters, tool policy and the tool server's own checks are all
exercised. Retrieval-only evals respect the user's labels too.

Skilvanta's 51 cases cover:

- facts and policies, including multi-step reasoning
- leaks of restricted documents
- follow-up questions, and a Hinglish question
- tool use: questions that need a tool, a tool error, and a question that must not use tools
- agent runs: multi-step lookups, a booking approved and one declined, a refund request
- access by role: what a learner, mentor, support agent and admin can each read and do, and
  a mentor refused another mentor's learner by the tool server
- prompt injection:
  - direct, role-play and Hinglish attempts
  - a question that merely uses the word "instructions"
  - injections planted in a document and in tool data

A `blocked` expectation checks whether a guardrail refused the message.

The latest run through the `claude_code` profile passed 50 of 51 in about 100 s. With prompt
caching, the same traffic would cost about $0.17 at API list prices. The one failing case is the Hinglish one:
the English-only reranker drops the section that holds the answer.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install
make check      # ruff, mypy, import-linter, pytest
make ingest     # build the document index (downloads a 67 MB embedding model once)
make run        # serves examples/edtech on http://127.0.0.1:8000
curl http://127.0.0.1:8000/healthz
TOKEN=$(make -s token as=riya)   # Skilvanta has access control on: requests need a token

# one JSON answer
curl http://127.0.0.1:8000/v1/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"stream": false, "messages": [{"role": "user", "content": "How do refunds work?"}]}'

# a question that needs a tool (the demo server acts as one synthetic learner)
curl http://127.0.0.1:8000/v1/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"stream": false, "messages": [{"role": "user", "content": "Which courses am I enrolled in?"}]}'

# an action: pauses for approval (see "Agent" above for the decisions call)
curl http://127.0.0.1:8000/v1/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"stream": false, "messages": [{"role": "user", "content": "Book my mentor slot on 29 November"}]}'

# streamed answer (Server-Sent Events: delta, tool, approval..., then done)
curl -N http://127.0.0.1:8000/v1/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Who are you?"}]}'

# structured output: a title for a conversation
curl http://127.0.0.1:8000/v1/title -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "How do refunds work?"}]}'
```

The example pack defaults to the `claude_code` profile. Without the Claude Code CLI, run
`BRAIN_MODEL_PROFILE=ollama make run` (with Ollama running) or set up a keyed provider.

## Licence

MIT
