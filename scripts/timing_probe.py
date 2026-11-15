"""One-epoch timing probe per task (LoRA and FT), written to results/probe/.

The measured seconds per epoch are scaled by the paper epoch counts in
scripts/estimate_compute.py to produce docs/compute_plan.md.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["rte", "mrpc", "stsb", "cola", "sst2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    args = ap.parse_args()
    out = ROOT / "results" / "probe"
    for task in args.tasks:
        for cfg in ("lora_roberta_base", "ft_roberta_base"):
            cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--config",
                   str(ROOT / "configs" / f"{cfg}.yaml"), "--task", task, "--seed", "0",
                   "--results-dir", str(out), "--ckpt-dir", str(ROOT / "checkpoints" / "probe"),
                   "--adapters-dir", str(ROOT / "checkpoints" / "probe_adapters"),
                   "--set", "epochs=1"]
            print(" ".join(cmd[1:]), flush=True)
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                print(f"probe failed for {cfg} {task} (rc={rc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
