"""Expansion of configs/matrix.yaml into an ordered list of runs."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .inject import canonical_targets

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Entry:
    group: str          # core | rank | proj
    method: str         # lora | ft
    task: str
    seed: int
    rank: int | None = None
    alpha: float | None = None
    targets: tuple = ()
    overrides: tuple = field(default_factory=tuple)

    @property
    def key(self) -> tuple:
        """Identity used to skip finished runs; shared between groups (e.g. core r=8 and rank)."""
        if self.method == "ft":
            return ("ft", self.task, "", "", "", str(self.seed))
        return ("lora", self.task, str(self.rank), _fmt(self.alpha), "+".join(self.targets),
                str(self.seed))

    @property
    def config(self) -> str:
        return "ft_config" if self.method == "ft" else "lora_config"


def _fmt(x) -> str:
    return "" if x is None else f"{float(x):g}"


def load_matrix(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "matrix.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _lora(group, task, seed, r, alpha, targets) -> Entry:
    return Entry(group, "lora", task, seed, r, alpha, tuple(canonical_targets(targets)))


def core_entries(m: dict) -> list[Entry]:
    c = m["core"]
    out = []
    for stage in c["stages"]:
        for task in stage["tasks"]:
            for seed in stage["seeds"]:
                for method in c["methods"]:
                    if method == "ft":
                        out.append(Entry("core", "ft", task, seed))
                    else:
                        lc = c["lora"]
                        out.append(_lora("core", task, seed, lc["r"], lc["alpha"], lc["targets"]))
    return out


def rank_entries(m: dict) -> list[Entry]:
    c = m["rank_ablation"]
    out = []
    for stage in c["stages"]:
        for task in stage["tasks"]:
            for seed in stage["seeds"]:
                for r in c["ranks"]:
                    out.append(_lora("rank", task, seed, r, r * c["alpha_over_r"], c["targets"]))
    return out


def proj_entries(m: dict) -> list[Entry]:
    c = m["projection_ablation"]
    out = []
    for task in c["tasks"]:
        for seed in c["seeds"]:
            for v in c["variants"]:
                out.append(_lora("proj", task, seed, v["r"], v["r"] * c["alpha_over_r"],
                                 v["targets"]))
    return out


def dedupe(entries: list[Entry]) -> list[Entry]:
    seen, out = set(), []
    for e in entries:
        if e.key not in seen:
            seen.add(e.key)
            out.append(e)
    return out


def build(name: str, m: dict | None = None) -> list[Entry]:
    m = m or load_matrix()
    parts = {"core": core_entries, "ablation": rank_entries, "proj_ablation": proj_entries}
    if name == "all":
        return dedupe(core_entries(m) + rank_entries(m) + proj_entries(m))
    return dedupe(parts[name](m))
