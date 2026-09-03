.PHONY: help install install-dev install-backend import-pool lint fmt test test-fast test-repro \
        public-pack registry-snapshot backend backend-api validator miner-validate \
        docker-sandbox-up docker-sandbox-down clean

PY ?= python3
PACK_DIR ?= data/public_pack

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime deps only)
	$(PY) -m pip install -e .

install-dev: ## Install with test + lint tooling
	$(PY) -m pip install -e ".[dev]"

install-backend: ## Install with backend evaluation dependencies
	$(PY) -m pip install -e ".[backend,dev]"

lint: ## Static checks
	ruff check capability_subnet neurons tests

fmt: ## Auto-format
	ruff check --fix capability_subnet neurons tests
	ruff format capability_subnet neurons tests

test: ## Full test suite
	$(PY) -m pytest -q

test-fast: ## Skip GPU / docker / chain marked tests
	$(PY) -m pytest -q -m "not gpu and not docker and not chain"

test-repro: ## Reconstruction determinism suite only
	$(PY) -m pytest -q tests/reproducibility

public-pack: ## Regenerate the public workflow pack from its published seed
	$(PY) -m capability_subnet.workflows.cli generate-public-pack --out $(PACK_DIR)

import-pool: ## Fetch and normalise the certified pool from its pinned upstream sources
	$(PY) scripts/import_public_adapters.py --out pool --write-registry

registry-snapshot: ## Print the frozen certified adapter snapshot digest
	$(PY) -m capability_subnet.registry.cli snapshot

validator: ## Run the thin validator neuron
	$(PY) neurons/validator.py

miner-validate: ## Validate a recipe file locally (RECIPE=path/to/recipe.json)
	$(PY) -m capability_subnet.miner.cli validate --recipe $(RECIPE)

docker-sandbox-up: ## Bring up the sandbox tool services for local debugging
	docker compose -f docker/docker-compose.sandbox.yml up -d

docker-sandbox-down: ## Tear the sandbox tool services down
	docker compose -f docker/docker-compose.sandbox.yml down -v

clean: ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
