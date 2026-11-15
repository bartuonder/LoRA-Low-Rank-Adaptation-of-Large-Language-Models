"""Start scripts/run_queue.py as a detached background process (Windows and POSIX).

The process keeps running after the launching terminal or agent session ends; re-running the
same command resumes the queue (finished runs are skipped, a live runner is detected by lock).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: launch_detached.py --matrix {core,ablation,proj_ablation,all}")
        return 2
    name = "queue_" + (args[args.index("--matrix") + 1] if "--matrix" in args else "x")
    (ROOT / "logs").mkdir(exist_ok=True)
    log = open(ROOT / "logs" / f"{name}.log", "a", encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_queue.py"), *args]
    kwargs = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "cwd": ROOT}
    if sys.platform == "win32":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_NO_WINDOW)
    else:
        kwargs["start_new_session"] = True
    p = subprocess.Popen(cmd, **kwargs)
    print(f"started queue runner pid {p.pid}; log: logs/{name}.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
