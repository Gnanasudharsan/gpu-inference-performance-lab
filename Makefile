PY      ?= .venv/bin/python
UV      ?= uv
.DEFAULT_GOAL := help

.PHONY: help install lint format typecheck test test-triton check env clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

install:  ## create .venv and install gpulab with dev extras
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e ".[dev]"

lint:  ## ruff lint + format check
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:  ## apply ruff formatting
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

typecheck:  ## mypy (strict defs) on the package
	$(PY) -m mypy

test:  ## unit + correctness tests on the devices available here
	$(PY) -m pytest

test-triton:  ## run Triton kernels through the CPU interpreter (Linux only; needs triton)
	TRITON_INTERPRET=1 $(PY) -m pytest -m triton

check: lint typecheck test  ## all local quality gates

env:  ## print provenance report
	$(PY) -m gpulab.cli env

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage results/scratch
