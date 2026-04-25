.PHONY: install install-dev test test-all lint type fmt clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -m "not gpu and not live_api"

test-gpu:
	pytest -m gpu

test-live:
	pytest -m live_api -s -v

test-all:
	pytest

lint:
	ruff check .

fmt:
	ruff check --fix .

type:
	mypy src/kvtrace

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov outputs/*.jsonl
