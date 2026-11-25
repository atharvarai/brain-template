# Make your own brain

A brain for your company is one folder, a **domain pack**. You write the folder; you do not
change `core/`. This guide builds a pack from scratch, in the order that gets you to a
working, measured assistant fastest. The two example packs are complete references:
`examples/rides` (small) and `examples/edtech` (full).

## 0. Before you start

- [uv](https://docs.astral.sh/uv/) and Python 3.12 or later. Run `make install`, then
  `make check`.
- A model. Any one of these works:
  - the Claude Code CLI on a subscription (no API key)
  - Ollama running locally
  - an Anthropic or OpenAI API key

  Section 2 shows how to switch between them.

## 1. Lay out the folder

```
packs/acme/
  brain.yaml            # everything configurable, in one file
  prompts/persona.v1.md # who the assistant is for your company
  docs/                 # your documents, as Markdown
  policy.yaml           # roles: which documents and tools each may use
  seed/users.yaml       # demo users, for local tokens and evals
  mcp_server/server.py  # optional stand-in for your backend's tools
  evals/cases.jsonl     # what good answers look like
```

The fastest start is copying the small example and editing it:
`cp -r examples/rides packs/acme`. Every command takes the pack as a variable:
`PACK=packs/acme make ingest`.

## 2. `brain.yaml`

Start with the parts you need. Every section except `name` and `models` is optional, and
the defaults are safe.

```yaml
name: Acme
description: Sells and services industrial anvils.

models:
  active: claude_code          # or ollama, anthropic, openai: a profile below
  profiles:
    claude_code:
      small: [{provider: claude_code, model: haiku}]
      mid:   [{provider: claude_code, model: sonnet}]
      large: [{provider: claude_code, model: opus}]
    anthropic:                 # a second entry in a list is the failover
      small: [{provider: anthropic, model: claude-haiku-5-5}]
      mid:   [{provider: anthropic, model: claude-sonnet-5-5},
              {provider: anthropic, model: claude-haiku-5-5}]
      large: [{provider: anthropic, model: claude-opus-5-5}]
  prices:                      # US dollars per million tokens; used for every cost figure
    claude-sonnet-5-5: {input: 2.00, output: 10.00, cache_read: 0.20, cache_write: 2.50}

rag: {docs: docs, top_k: 5}
evals: {cases: evals/cases.jsonl, min_pass_rate: 0.85}
```

Code asks for a tier (small, mid, large), never a model, so switching vendor means
changing `models.active`, or setting `BRAIN_MODEL_PROFILE` for one process.

## 3. The persona

`prompts/persona.v1.md` says who the assistant works for and how it should sound. The core
system prompt around it already handles the rules: grounding, citations, refusing to leak
instructions, treating documents and tool results as data, approvals and language.

> **Do not fight the core rules.** Zipvaha's first persona said "drivers write in Hinglish,
> reply in the same mix", and the model answered English questions in Hinglish. Say what is
> specific to your company and let the core handle the rest.

Every prompt is versioned (`persona.v2.md` next to `v1`). The newest version is used unless
`prompts.pinned` says otherwise, so a rollback is one line, even for core prompts:
`pinned: {system: 4}`. To override a core prompt entirely, put a file with the same name
in your `prompts/` folder.

## 4. Documents

- **Write Markdown with `##` headings.** Retrieval splits on headings, and eval cases can
  name a section (`refunds.md#Refund amounts`), so headings are worth getting right.
- **Restrict documents with labels.** Front matter like `labels: [internal]` hides a
  document from everyone whose roles do not grant that label. Unlabelled documents are
  public.
- **Build and check the index.** Run `PACK=packs/acme make ingest`, then
  `make search q="..."` to see what a question retrieves. Re-run ingest whenever documents
  change; the answer cache notices and drops stale entries.

The default embedding and reranking models are English-only. For other languages, set
multilingual models under `rag.embedder` and `rag.reranker`, and use
`make eval-retrieval` to see the difference.

## 5. Access

Your backend signs a JWT for each user and sends it with every request to the brain:

```json
{"sub": "u-123", "roles": ["member"], "iss": "acme-backend", "aud": "acme-brain", "exp": 1767225600}
```

```yaml
access:
  jwt: {algorithm: RS256, public_key_file: keys/backend.pem,
        issuer: acme-backend, audience: acme-brain}
  policy: policy.yaml
  dev_users: seed/users.yaml   # local only: lets `make token as=...` sign test tokens (HS256)
```

`policy.yaml` maps roles to readable labels and callable tools. Anything not listed is
denied:

```yaml
roles:
  member: {read: [], tools: [lookup_order]}
  staff:  {read: [internal], tools: [lookup_order, refund_order]}
```

## 6. Tools

Tools come from MCP servers that your backend runs. The brain discovers them at start-up
and calls them as the user, forwarding the user's token. Your server stays in charge of
its own permissions.

- **Write the description for the model.** Say exactly what a tool returns. A Skilvanta
  tool was never called for "mentor notes" until its description mentioned them.
- **Mark tools that change nothing with `readOnlyHint`.** Anything else waits for the
  user's approval before it runs. That is the safe default if you forget.
- **Check the token yourself and apply your own rules.** The brain's policy is the first
  gate, not the only one.
- **Connect over HTTP in production, stdio locally:**
  `acme: {transport: http, url: "https://backend.internal/mcp"}`.

Check what the brain sees with `make tools` and `make tools tool=lookup_order args='{...}' as=alice`.

## 7. Evals

One case per line in `evals/cases.jsonl`. A case runs as a demo user and lists what a good
answer shows:

```json
{"id": "return-window", "question": "How long do I have to return an anvil?",
 "expect": {"contains_any": ["30 days"], "retrieves": ["returns.md#Window"],
            "rubric": "States the 30-day window and nothing contradicting it."}}
```

| Check | Use it for |
|---|---|
| `contains_all`, `contains_any` | facts that must appear (numbers are matched without thousands separators) |
| `excludes` | text that must never appear, such as leaked internal figures. It also matches negations ("it is **not** Rs 3,400"), so use a rubric for claims |
| `retrieves`, `cites` | the right document or section was found, or cited |
| `rubric` | anything else, graded by a judge model |
| `calls`, `asks_approval` | which tools ran, which actions paused (`[]` means none) |
| `blocked` | whether a guardrail must refuse the message |

Cover each role (`"user": "staff"`), what each must not see, one injection attempt, and a
question outside your scope.

```bash
PACK=packs/acme make eval-retrieval   # no model, seconds: put it in CI
PACK=packs/acme make eval             # every check, with the model: before each change
```

Every run is saved and compared with the previous one, listing regressions and fixes.
Change one thing at a time, and believe the evals over your intuition: in this repo they
overturned a router, a semantic cache and a tool protocol that all looked fine.

## 8. Run it

```bash
PACK=packs/acme make run              # http://127.0.0.1:8000
open http://127.0.0.1:8000/demo/      # the demo chat page; paste a token from:
PACK=packs/acme make token as=alice
```

`make traces` and `make audit` show what happened, and why.

## 9. Before production

- **Keys:** use RS256 with your backend's public key, not a shared HS256 secret. Never ship
  a `dev_users` file or the Makefile's development secret.
- **Text in traces:** set `observability.capture_content: false` unless you may store user
  text. Personal data is redacted either way: set the `guardrails.pii.kinds` your users'
  data contains.
- **Tool servers:** connect over HTTP, with your own authentication.
- **Semantic cache:** turn it on only after `make cache-check` shows a threshold with no
  false hits on your own question pairs.
- **Limits:** set `agent` budgets and `approval_ttl_minutes` for your traffic.
- **CI:** gate on `eval-retrieval`, and run `make eval` before every prompt or model change.
- **Still to build (see [PLAN.md](PLAN.md)):** a Postgres or pgvector store, OpenTelemetry
  export, token revocation and rate limits.
