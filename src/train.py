"""Train one (config, task, seed) run. Used for both LoRA and full fine-tuning.

Example:
    python src/train.py --config configs/lora_roberta_base.yaml --task rte --seed 0 \
        --set lora.r=4 lora.alpha=4
"""
import argparse
import sys
from pathlib import Path

import torch

from lora.config import ROOT, load_config
from lora.trainer import run_training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="dotted config overrides, e.g. lora.r=4 epochs=1")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--adapters-dir", default=str(ROOT / "adapters"),
                    help="best-epoch LoRA A/B + head weights go to <dir>/<task>/seed<k>/")
    ap.add_argument("--data-limit", type=int, default=None, help="use only the first N examples")
    args = ap.parse_args()

    cfg = load_config(args.config, args.task, args.set)
    try:
        run_training(cfg, args.seed, Path(args.results_dir), Path(args.ckpt_dir),
                     data_limit=args.data_limit, adapters_dir=Path(args.adapters_dir))
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory", file=sys.stderr)
        return 3  # the queue runner retries with gradient accumulation
    return 0


if __name__ == "__main__":
    sys.exit(main())
