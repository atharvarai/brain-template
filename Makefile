PACK ?= examples/edtech

.PHONY: install run ingest search eval eval-retrieval traces test lint check

install:
	uv sync

run:
	BRAIN_PACK=$(PACK) uv run uvicorn core.api.app:create_app --factory --reload

ingest:
	BRAIN_PACK=$(PACK) uv run python -m core.cli ingest

# usage: make search q="how do refunds work"
search:
	BRAIN_PACK=$(PACK) uv run python -m core.cli search "$(q)"

# Ask the model every eval case and grade the answers (uses the active model profile).
eval:
	BRAIN_PACK=$(PACK) uv run python -m core.cli eval

# Retrieval checks only: no model needed, so CI can run it.
eval-retrieval:
	BRAIN_PACK=$(PACK) uv run python -m core.cli eval --retrieval

traces:
	BRAIN_PACK=$(PACK) uv run python -m core.cli traces

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run lint-imports

check: lint test
