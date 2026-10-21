.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python3

.PHONY: ej1 ej2 clean help

ej1: ## Sync dependencies and run Exercise 1
	$(UV) sync --package ej1
	$(UV) run ej1

ej2: ## Sync dependencies and run Exercise 2
	$(UV) sync --package ej2
	$(UV) run ej2

clean: ## Clean cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .ruff_cache .pytest_cache .mypy_cache build dist *.egg-info

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
