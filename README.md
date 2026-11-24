# Brain Template

An industry-agnostic template for a company's AI microservice: its "brain".

Clone it, add one folder that describes your company (docs, tools, roles, prompts, eval
cases), and you get an assistant built from clearly separated AI layers: model, prompts,
RAG, tools (MCP), agent, access control, guardrails, cache and router, observability and evals.

**Status: early.** Milestone M0 (skeleton) is done. See [docs/PLAN.md](docs/PLAN.md) for the
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
`brain.yaml` maps each tier to an ordered fallback chain of `{provider, model}` entries, and
providers are an enum (`anthropic`, `openai`, `ollama`, `claude_code`, `fake`). Switching
vendor is a config change.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install
make check      # ruff, mypy, import-linter, pytest
make run        # serves examples/edtech on http://127.0.0.1:8000
curl http://127.0.0.1:8000/healthz
```

## Licence

MIT
