.PHONY: lint typecheck security test all

lint:
	uv run --isolated ruff format --check .
	uv run --isolated ruff check .

typecheck:
	uv run --isolated mypy src tests

security:
	uv run --isolated bandit -c pyproject.toml -r src
	uv run --isolated pip-audit
	detect-secrets scan --exclude-files 'uv\.lock' > .secrets.baseline

test:
	uv run --isolated pytest

all: lint typecheck security test
