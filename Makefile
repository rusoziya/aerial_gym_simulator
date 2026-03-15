# Aerial Gym Simulator — Common Commands
# Usage: make train CONFIG=configs/train_gate_navigation.yaml

CONFIG ?= configs/train_gate_navigation.yaml
PYTHON ?= python3

# ─── Training & Evaluation ───────────────────────────────────────
.PHONY: train eval play validate-config dry-run

train:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --set mode=train

eval:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --set mode=eval

play:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --set mode=play

validate-config:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --validate-only

dry-run:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --dry-run

# ─── Quick Targets (common configurations) ───────────────────────
.PHONY: train-gate train-position eval-gate

train-gate:
	$(PYTHON) -m aerial_gym.run --config configs/train_gate_navigation.yaml

train-position:
	$(PYTHON) -m aerial_gym.run --config configs/train_position_setpoint.yaml

eval-gate:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_navigation.yaml

# ─── Development ─────────────────────────────────────────────────
.PHONY: lint format test pre-commit

lint:
	ruff check aerial_gym/ tests/

format:
	ruff format aerial_gym/ tests/ scripts/

test:
	$(PYTHON) -m pytest tests/ -v --tb=short || test $$? -eq 5

pre-commit:
	pre-commit run --all-files

# ─── Environment Setup ──────────────────────────────────────────
.PHONY: install install-dev

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# ─── Cleanup ────────────────────────────────────────────────────
.PHONY: clean

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: help
help:
	@echo "Aerial Gym Simulator"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train CONFIG=<yaml>    Train with config file"
	@echo "  make eval CONFIG=<yaml>     Evaluate with config file"
	@echo "  make play CONFIG=<yaml>     Play/visualize with config file"
	@echo "  make dry-run CONFIG=<yaml>  Show command without executing"
	@echo "  make validate-config        Validate config file"
	@echo ""
	@echo "Quick Targets:"
	@echo "  make train-gate             Train gate navigation (SF)"
	@echo "  make train-position         Train position setpoint (rl_games)"
	@echo "  make eval-gate              Evaluate gate navigation"
	@echo ""
	@echo "Development:"
	@echo "  make lint                   Run ruff linter"
	@echo "  make format                 Format code with ruff"
	@echo "  make test                   Run tests"
	@echo "  make pre-commit             Run all pre-commit hooks"
	@echo ""
	@echo "Setup:"
	@echo "  make install                Install package"
	@echo "  make install-dev            Install with dev deps + pre-commit"
	@echo "  make clean                  Remove __pycache__ and .pyc files"
