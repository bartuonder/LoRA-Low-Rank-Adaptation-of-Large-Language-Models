"""Queue runner: executes the run matrix one job at a time and resumes after interruptions.

Skips runs already present in <results-dir>/runs.csv, retries a failed run up to 2 times
(halving the micro-batch after an out-of-memory error, so the effective batch size is kept),
records permanent failures in <results-dir>/failures.csv and moves on.

Start detached (Windows): python scripts/launch_detached.py --matrix core
Stop gracefully: create <results-dir>/STOP (the current run finishes first).
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from lora.matrix import Entry, build, load_matrix  # noqa: E402

MAX_RETRIES = 2
OOM_RC = 3


def done_keys(results_dir: Path) -> set:
    path = results_dir / "runs.csv"
    keys = set()
    if not path.exists():
        return keys
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] not in ("ok", "diverged"):
                continue
            if r["method"] == "ft":
                keys.add(("ft", r["task"], "", "", "", r["seed"]))
            else:
                keys.add(("lora", r["task"], str(int(float(r["rank"]))), f"{float(r['alpha']):g}",
                          r["targets"], r["seed"]))
    return keys


def batch_size_for(e: Entry, matrix: dict) -> int:
    cfg = yaml.safe_load((ROOT / matrix[e.config]).read_text(encoding="utf-8"))
    return cfg["tasks"][e.task]["batch_size"]


def command(e: Entry, matrix: dict, results_dir: Path, micro_batch: int | None) -> list[str]:
    cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--config",
           str(ROOT / matrix[e.config]), "--task", e.task, "--seed", str(e.seed),
           "--results-dir", str(results_dir), "--ckpt-dir", str(ROOT / "checkpoints")]
    sets = []
    if e.method == "lora":
        sets += [f"lora.r={e.rank}", f"lora.alpha={e.alpha:g}",
                 "lora.targets=[" + ",".join(e.targets) + "]"]
    if micro_batch:
        sets.append(f"micro_batch_size={micro_batch}")
    sets += list(e.overrides)
    return cmd + (["--set", *sets] if sets else [])


def record_failure(results_dir: Path, e: Entry, rc: int, attempts: int) -> None:
    path = results_dir / "failures.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "group", "method", "task", "rank", "targets", "seed", "rc",
                        "attempts"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), e.group, e.method, e.task, e.rank,
                    "+".join(e.targets), e.seed, rc, attempts])


def acquire_lock(results_dir: Path) -> Path:
    lock = results_dir / ".queue.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            if psutil.pid_exists(pid) and pid != os.getpid():
                print(f"another queue runner is active (pid {pid}); exiting")
                sys.exit(0)
        except ValueError:
            pass
    lock.write_text(str(os.getpid()))
    return lock


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, choices=["core", "ablation", "proj_ablation", "all"])
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_matrix()
    entries = build(args.matrix, matrix)
    todo = [e for e in entries if e.key not in done_keys(results_dir)]
    print(f"{len(entries)} runs in matrix '{args.matrix}', {len(todo)} pending")
    if args.dry_run:
        for e in todo:
            print(" ".join(command(e, matrix, results_dir, None)[2:]))
        return 0

    lock = acquire_lock(results_dir)
    stop = results_dir / "STOP"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    launched = 0
    try:
        for e in todo:
            if stop.exists() or (args.max_runs and launched >= args.max_runs):
                break
            if e.key in done_keys(results_dir):
                continue
            free_gb = shutil.disk_usage(ROOT).free / 2**30
            if free_gb < 5:
                print(f"only {free_gb:.1f} GB free; stopping")
                break
            bs = batch_size_for(e, matrix)
            micro, rc = None, 1
            name = f"{e.method}_{e.task}_r{e.rank}_{'-'.join(t[0] for t in e.targets)}_s{e.seed}"
            for attempt in range(1, MAX_RETRIES + 2):
                print(f"{time.strftime('%F %T')} start {name} attempt {attempt}", flush=True)
                with open(logs / f"{name}.log", "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== attempt {attempt} micro_batch={micro} ===\n")
                    lf.flush()
                    rc = subprocess.call(command(e, matrix, results_dir, micro), stdout=lf,
                                         stderr=subprocess.STDOUT, cwd=ROOT)
                if rc == 0:
                    break
                if rc == OOM_RC:
                    micro = max(1, (micro or bs) // 2)
            else:
                record_failure(results_dir, e, rc, MAX_RETRIES + 1)
                print(f"{time.strftime('%F %T')} FAILED {name} rc={rc}", flush=True)
            launched += 1
    finally:
        lock.unlink(missing_ok=True)
    print("queue finished" if not stop.exists() else "stopped by STOP file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
