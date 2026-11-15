"""Print the state of the queue: runner liveness, finished runs per group, failures, ETA."""
import csv
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lora.matrix import build  # noqa: E402
from run_queue import done_keys  # noqa: E402


def main() -> int:
    res = ROOT / "results"
    lock = res / ".queue.lock"
    alive = False
    if lock.exists():
        try:
            alive = psutil.pid_exists(int(lock.read_text().strip()))
        except ValueError:
            pass
    print(f"runner alive: {alive}")
    done = done_keys(res)
    total_left_s = 0.0
    secs = {}
    path = res / "runs.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                secs.setdefault((r["method"], r["task"]), []).append(float(r["train_time_s"]))
    for g in ("core", "ablation", "proj_ablation"):
        ent = build(g)
        left = [e for e in ent if e.key not in done]
        print(f"{g:14s} {len(ent) - len(left):3d}/{len(ent)} finished")
        for e in left:
            s = secs.get((e.method, e.task))
            total_left_s += (sum(s) / len(s)) if s else 0
    fail = res / "failures.csv"
    if fail.exists():
        print("failures:\n" + fail.read_text(encoding="utf-8"))
    print(f"rough remaining training time for runs with a measured analogue: "
          f"{total_left_s / 3600:.1f} h")
    free_gb = psutil.disk_usage(str(ROOT)).free / 2**30
    print(f"free disk: {free_gb:.0f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
