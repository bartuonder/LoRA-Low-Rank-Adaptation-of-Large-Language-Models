PY ?= $(firstword $(wildcard .venv/Scripts/python.exe .venv/bin/python) python)
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128

.PHONY: setup test lint smoke probe core ablation proj_ablation bench report all \
        core-detached ablation-detached proj_ablation-detached paper

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url $(TORCH_INDEX)
	$(PY) -m pip install -e ".[dev]"

paper:
	$(PY) scripts/fetch_paper.py
	$(PY) scripts/verify_paper_numbers.py

lint:
	$(PY) -m ruff check .

test: lint
	$(PY) -m pytest -q

# Tiny end-to-end runs (a few minutes): LoRA and full fine-tuning on a slice of RTE.
smoke:
	$(PY) src/train.py --config configs/lora_roberta_base.yaml --task rte --seed 0 \
		--set epochs=2 --data-limit 256 --results-dir outputs/smoke --ckpt-dir outputs/smoke/ckpt --adapters-dir outputs/smoke/adapters
	$(PY) src/train.py --config configs/ft_roberta_base.yaml --task rte --seed 0 \
		--set epochs=2 --data-limit 256 --results-dir outputs/smoke --ckpt-dir outputs/smoke/ckpt --adapters-dir outputs/smoke/adapters

# One-epoch timing probe per task, used for docs/compute_plan.md.
probe:
	$(PY) scripts/timing_probe.py

# Training targets go through the queue runner; re-running a target resumes it.
core:
	$(PY) scripts/run_queue.py --matrix core

ablation:
	$(PY) scripts/run_queue.py --matrix ablation

proj_ablation:
	$(PY) scripts/run_queue.py --matrix proj_ablation

core-detached:
	$(PY) scripts/launch_detached.py --matrix core

ablation-detached:
	$(PY) scripts/launch_detached.py --matrix ablation

proj_ablation-detached:
	$(PY) scripts/launch_detached.py --matrix proj_ablation

# Run only while no training job is active.
bench:
	$(PY) src/bench.py

report:
	$(PY) scripts/make_report.py

all: test smoke core ablation proj_ablation bench report
