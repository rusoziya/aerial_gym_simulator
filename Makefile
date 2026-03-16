# Aerial Gym Simulator — Sample Factory Training & Evaluation
# Usage: make train CONFIG=configs/train_gate_sf.yaml

CONFIG ?= configs/train_gate_sf.yaml
CONDA_PREFIX ?= $(HOME)/miniforge3/envs/aerialgym
PYTHON ?= LD_LIBRARY_PATH=$(CONDA_PREFIX)/lib $(CONDA_PREFIX)/bin/python

# ─── Training & Evaluation ───────────────────────────────────────
.PHONY: train eval play validate-config dry-run

train:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --set mode=train --log

eval:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --set mode=eval

play:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --set mode=play

validate-config:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --validate-only

dry-run:
	$(PYTHON) -m aerial_gym.run --config $(CONFIG) --dry-run

# ─── Quick Targets ───────────────────────────────────────────────
.PHONY: train-gate train-gate-fixed train-gate-arc train-gate-dynamic
.PHONY: eval-gate-drone-only eval-gate-dynamic eval-gate-sweeping
.PHONY: eval-gate-arc eval-gate-locked eval-gate-static eval-all-modalities

train-gate:
	$(PYTHON) -m aerial_gym.run --config configs/train_gate_sf.yaml

train-gate-fixed:
	$(PYTHON) -m aerial_gym.run --config configs/train_gate_sf_fixed_orient.yaml

train-gate-arc:
	$(PYTHON) -m aerial_gym.run --config configs/train_gate_sf_arc_follow.yaml

train-gate-dynamic:
	$(PYTHON) -m aerial_gym.run --config configs/train_gate_sf_dynamic_follow.yaml

eval-gate-drone-only:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_drone_only.yaml

eval-gate-dynamic:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_dynamic_follow.yaml

eval-gate-sweeping:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_sweeping.yaml

eval-gate-arc:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_arc_follow.yaml

eval-gate-locked:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_locked_yaw.yaml

eval-gate-static:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_static_random.yaml

eval-all-modalities:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_all_modalities.yaml

# ─── Isaac Lab Backend ───────────────────────────────────────────
.PHONY: train-gate-lab eval-gate-lab

train-gate-lab:
	$(PYTHON) -m aerial_gym.run --config configs/train_gate_sf_isaaclab.yaml --log

eval-gate-lab:
	$(PYTHON) -m aerial_gym.run --config configs/eval_gate_drone_only.yaml --set common.backend=isaaclab

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
	@echo "Aerial Gym Simulator — Sample Factory Pipeline"
	@echo ""
	@echo "Training:"
	@echo "  make train CONFIG=<yaml>      Train with config file"
	@echo "  make train-gate               Gate nav (default SF config)"
	@echo "  make train-gate-fixed         Gate nav, fixed orientation"
	@echo "  make train-gate-arc           Gate nav, arc-follow camera"
	@echo "  make train-gate-dynamic       Gate nav, dynamic-follow camera"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval CONFIG=<yaml>       Evaluate with config file"
	@echo "  make eval-gate-drone-only     Drone-only ablation"
	@echo "  make eval-gate-dynamic        Dynamic-follow eval"
	@echo "  make eval-gate-sweeping       Yaw-sweep eval"
	@echo "  make eval-gate-arc            Arc-follow eval"
	@echo "  make eval-gate-locked         Locked-yaw eval"
	@echo "  make eval-gate-static         Static-random eval"
	@echo "  make eval-all-modalities      All modalities x seeds x levels"
	@echo ""
	@echo "Utilities:"
	@echo "  make dry-run CONFIG=<yaml>    Show command without executing"
	@echo "  make validate-config          Validate config file"
	@echo "  make play CONFIG=<yaml>       Visualize with config file"
	@echo ""
	@echo "Development:"
	@echo "  make lint                     Run ruff linter"
	@echo "  make format                   Format code with ruff"
	@echo "  make test                     Run tests"
	@echo "  make pre-commit               Run all pre-commit hooks"
	@echo "  make install-dev              Install with dev deps"
	@echo "  make clean                    Remove __pycache__ and .pyc"
