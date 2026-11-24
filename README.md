# Brain Template

An industry-agnostic template for a company's AI microservice: its "brain".

Clone it, add one folder that describes your company (docs, tools, roles, prompts, eval
cases), and you get an assistant built from clearly separated AI layers: model, prompts,
RAG, tools (MCP), agent, access control, guardrails, cache and router, observability and evals.

**Status: early.** Milestones M0 (skeleton) and M1 (model layer) are done: the service
answers chat requests through a vendor-neutral model layer with failover. There is no
retrieval, tool use or access control yet. See [docs/PLAN.md](docs/PLAN.md) for the design
and the milestone list.

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

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install
make check      # ruff, mypy, import-linter, pytest
make run        # serves examples/edtech on http://127.0.0.1:8000
curl http://127.0.0.1:8000/healthz

# one JSON answer
curl http://127.0.0.1:8000/v1/chat -H 'content-type: application/json' \
  -d '{"stream": false, "messages": [{"role": "user", "content": "Who are you?"}]}'

# streamed answer (Server-Sent Events: delta..., then done)
curl -N http://127.0.0.1:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Who are you?"}]}'
```

The example pack defaults to the `claude_code` profile. Without the Claude Code CLI, run
`BRAIN_MODEL_PROFILE=ollama make run` (with Ollama running) or set up a keyed provider.

## Licence

MIT
