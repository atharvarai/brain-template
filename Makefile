PACK ?= examples/edtech

.PHONY: install run test lint check

install:
	uv sync

run:
	BRAIN_PACK=$(PACK) uv run uvicorn core.api.app:create_app --factory --reload

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run lint-imports

check: lint test
