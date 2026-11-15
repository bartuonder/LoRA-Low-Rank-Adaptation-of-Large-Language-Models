import copy
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _coerce(value: str):
    out = yaml.safe_load(value)
    if isinstance(out, str):  # PyYAML reads "1e-3" (no dot) as a string
        try:
            return float(out)
        except ValueError:
            pass
    return out


def set_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, task: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, flatten the chosen task into it and apply `key=value` overrides.

    Overrides use dotted keys, e.g. `lora.r=4`, `lr=1e-4`, `epochs=2`. Task-level keys
    (lr, epochs, batch_size, metric, glue_name) are addressed without prefix.
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tasks = cfg.pop("tasks")
    if task not in tasks:
        raise KeyError(f"task {task!r} not in {sorted(tasks)}")
    t = tasks[task]
    if t.get("enabled", True) is False:
        raise ValueError(f"task {task!r} is configured but out of scope for this project")
    cfg["task"] = task
    cfg.update({k: v for k, v in t.items() if k != "enabled"})
    for item in overrides or []:
        key, _, val = item.partition("=")
        set_path(cfg, key.strip(), _coerce(val))
    if cfg["method"] == "lora":
        cfg["lora"]["targets"] = list(cfg["lora"]["targets"])
    return cfg


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_id(cfg: dict, seed: int) -> str:
    if cfg["method"] == "lora":
        lc = cfg["lora"]
        tg = "".join(t[0] for t in lc["targets"])
        return f"lora_{cfg['task']}_r{lc['r']}_{tg}_s{seed}"
    return f"ft_{cfg['task']}_s{seed}"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def deepcopy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
