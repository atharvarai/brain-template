PACK ?= examples/edtech
# Signs and checks demo user tokens. A development-only value: set your own in production.
BRAIN_JWT_SECRET ?= dev-only-secret-not-for-production-0001
export BRAIN_JWT_SECRET

.PHONY: install run ingest search eval eval-retrieval traces tools mcp-server token audit cache-check test lint check

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

# List the tools discovered from the pack's MCP servers.
# Call one as a demo user: make tools tool=get_course_progress args='{"course": "BEPY"}' as=riya
tools:
	BRAIN_PACK=$(PACK) uv run python -m core.cli tools $(if $(tool),--call $(tool) --args '$(or $(args),{})') $(if $(as),--as $(as))

# Sign a token for a demo user: make token as=riya
token:
	@BRAIN_PACK=$(PACK) uv run python -m core.cli token $(or $(as),riya)

audit:
	BRAIN_PACK=$(PACK) uv run python -m core.cli audit

# Semantic cache: hits and wrong answers per similarity threshold, on the pack's pairs.
cache-check:
	BRAIN_PACK=$(PACK) uv run python -m core.cli cache-check

# Run the pack's demo MCP server over HTTP, for packs configured with transport: http.
mcp-server:
	cd $(PACK) && uv run python mcp_server/server.py --http --port 8792

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run lint-imports

check: lint test
