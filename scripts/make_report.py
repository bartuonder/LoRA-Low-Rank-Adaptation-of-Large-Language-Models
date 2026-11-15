"""Build all tables, figures and README blocks from results/ (nothing is typed by hand).

Inputs : results/runs.csv, results/bench.json, results/probe/runs.csv, paper/paper_results.yaml,
         docs/reasons.yaml, configs/*.yaml
Outputs: results/tables/*.md, figures/*.png, and the marked blocks inside README.md
         (<!-- BEGIN:name --> ... <!-- END:name -->).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lora.matrix import build, load_matrix  # noqa: E402

RESULTS = ROOT / "results"
FIG = ROOT / "figures"
TABLES = RESULTS / "tables"

# (task id, paper/table name, metric label); order used in the tables
TASKS = [("sst2", "SST-2", "Accuracy"), ("mrpc", "MRPC", "Accuracy"),
         ("cola", "CoLA", "Matthews corr."), ("rte", "RTE", "Accuracy"),
         ("stsb", "STS-B", "Pearson corr.")]
NAME = {t: n for t, n, _ in TASKS}
METRIC = {t: m for t, _, m in TASKS}
SMALL_FIRST = ["rte", "mrpc", "stsb", "cola", "sst2"]
CORE_TARGETS = "query+value"
RANKS = [1, 2, 4, 8]

# validated categorical slots (light surface), fixed order
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


# ---------------------------------------------------------------- data helpers
def load_runs() -> pd.DataFrame:
    path = RESULTS / "runs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"targets": str, "alpha": str})
    df = df[df["status"].isin(["ok", "diverged"])].drop_duplicates("run_id", keep="last")
    df["curve"] = df["epoch_metrics"].apply(json.loads)
    return df


def pick(df, method, task, rank=None, targets=None) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["method"] == method) & (df["task"] == task)
    if method == "lora":
        if rank is not None:
            m &= df["rank"] == rank
        if targets is not None:
            m &= df["targets"] == targets
    return df[m].sort_values("seed")


def stat(x) -> dict | None:
    a = np.asarray(list(x), dtype=float)
    if len(a) == 0:
        return None
    return {"n": len(a), "mean": a.mean(), "std": a.std(ddof=1) if len(a) > 1 else float("nan"),
            "median": float(np.median(a)), "min": a.min(), "max": a.max()}


def f1(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"


def pm(s) -> str:
    return "–" if s is None else f"{f1(s['mean'])} ± {f1(s['std'])}"


def signed(x) -> str:
    return "–" if x is None else f"{x:+.1f}"


def md_table(headers, rows, align=None) -> str:
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def best_scores(df, **kw) -> list[float]:
    return list(pick(df, **kw)["best_metric"]) if not df.empty else []


def last_scores(df, **kw) -> list[float]:
    return list(pick(df, **kw)["final_metric"]) if not df.empty else []


def paper_cell(paper, row, task) -> dict:
    return paper["roberta_base"][row]["scores"][NAME[task]]


def fmt_paper(c) -> str:
    return f"{c['value']:.1f}" + (f" ± {c['std']:.1f}" if "std" in c else "")


# ---------------------------------------------------------------- blocks
def block_paper_vs_mine(df, paper, reasons) -> tuple[str, dict]:
    rows, info = [], {}
    meds, pvals = [], []
    for t, name, metric in TASKS:
        s = stat(best_scores(df, method="lora", task=t, rank=8, targets=CORE_TARGETS))
        pc = paper_cell(paper, "lora", t)
        delta = None if s is None else s["median"] - pc["value"]
        if s is not None:
            meds.append(s["median"])
            pvals.append(pc["value"])
        spread = pc.get("std", 0.0)
        kind = "close" if delta is not None and abs(delta) <= spread else (
            "lower" if (delta or 0) < 0 else "higher")
        reason = reasons["tasks"][t][kind] if s is not None else "Not run."
        info[t] = {"stat": s, "paper": pc, "delta": delta}
        rows.append([name, metric, fmt_paper(pc), "n/a", pm(s),
                     f1(s["median"]) if s else "–", signed(delta),
                     s["n"] if s else 0, reason])
    if len(meds) == len(TASKS):
        d = float(np.mean(meds) - np.mean(pvals))
        kind = "close" if abs(d) <= 0.5 else ("lower" if d < 0 else "higher")
        # the 0.5 tolerance only selects the wording of the hypothesis
        rows.append(["Mean of the five tasks", "–", f1(float(np.mean(pvals))), "n/a", "–",
                     f1(float(np.mean(meds))), signed(d), "–", reasons["average"][kind]])
    headers = ["Task", "Metric", "Paper (published)", "Paper (dagger row)", "Mine: mean ± std",
               "Mine: median", "Δ (median − paper)", "Seeds", "Likely reason (hypothesis)"]
    return md_table(headers, rows, ["l", "l", "r", "r", "r", "r", "r", "r", "l"]), info


def block_core(df) -> str:
    rows = []
    for t, name, _ in TASKS:
        for method, label in (("lora", "LoRA (r=8, Wq+Wv)"), ("ft", "Full fine-tuning")):
            kw = {"method": method, "task": t}
            if method == "lora":
                kw.update(rank=8, targets=CORE_TARGETS)
            b, last = stat(best_scores(df, **kw)), stat(last_scores(df, **kw))
            sub = pick(df, **kw)
            tp = (int(sub["trainable_params"].iloc[0]) - int(sub["head_params"].iloc[0])
                  if len(sub) else None)
            tp = int(sub["trainable_params"].iloc[0]) if (len(sub) and method == "ft") else tp
            rows.append([name, label, f"{tp:,}" if tp else "–", pm(b),
                         f1(b["median"]) if b else "–",
                         f"{f1(b['min'])} – {f1(b['max'])}" if b else "–", pm(last),
                         b["n"] if b else 0])
    h = ["Task", "Method", "Trainable params (excl. head)", "Best epoch: mean ± std",
         "Best epoch: median", "Min – max", "Last epoch: mean ± std", "Seeds"]
    return md_table(h, rows, ["l", "l", "r", "r", "r", "r", "r", "r"])


def block_ft(df, paper) -> str:
    rows = []
    for t, name, _ in TASKS:
        ft = stat(best_scores(df, method="ft", task=t))
        lo = stat(best_scores(df, method="lora", task=t, rank=8, targets=CORE_TARGETS))
        pft = paper_cell(paper, "ft", t)["value"]
        plo = paper_cell(paper, "lora", t)["value"]
        mine_gap = None if not (ft and lo) else lo["median"] - ft["median"]
        rows.append([name, f"{pft:.1f}", pm(ft), f1(ft["median"]) if ft else "–",
                     fmt_paper(paper_cell(paper, "lora", t)), f1(lo["median"]) if lo else "–",
                     signed(plo - pft), signed(mine_gap)])
    h = ["Task", "Paper FT (Liu et al.)", "My FT: mean ± std", "My FT: median", "Paper LoRA",
         "My LoRA: median", "Paper: LoRA − FT", "Mine: LoRA − FT (medians)"]
    return md_table(h, rows)


def block_rank(df) -> tuple[str, str]:
    mean_rows, med_rows = [], []
    for r in RANKS:
        mrow, drow = [f"r = {r}"], [f"r = {r}"]
        for t, _, _ in TASKS:
            s = stat(best_scores(df, method="lora", task=t, rank=r, targets=CORE_TARGETS))
            mrow.append(pm(s) + (f" (n={s['n']})" if s and s["n"] != 5 else ""))
            drow.append(f1(s["median"]) if s else "–")
        lora_params = None
        sub = df[(df["method"] == "lora") & (df["rank"] == r) & (df["targets"] == CORE_TARGETS)] \
            if not df.empty else df
        if len(sub):
            lora_params = int(sub["trainable_params"].iloc[0]) - int(sub["head_params"].iloc[0])
        mrow.insert(1, f"{lora_params:,}" if lora_params else "–")
        drow.insert(1, f"{lora_params:,}" if lora_params else "–")
        mean_rows.append(mrow)
        med_rows.append(drow)
    h = ["Rank", "LoRA params (excl. head)"] + [NAME[t] for t, _, _ in TASKS]
    return md_table(h, mean_rows), md_table(h, med_rows)


def proj_variants() -> list[dict]:
    m = load_matrix()
    out = []
    for v in m["projection_ablation"]["variants"]:
        tg = "+".join(v["targets"])
        label = "+".join({"query": "Wq", "key": "Wk", "value": "Wv", "output": "Wo"}[x]
                         for x in v["targets"])
        out.append({"name": v["name"], "label": label, "targets": tg, "r": v["r"]})
    return out


def block_proj(df) -> str:
    tasks = ["rte", "mrpc", "stsb"]
    rows = []
    for v in proj_variants():
        row = [v["label"] + (" (paper default)" if v["name"] == "qv" else ""), v["r"]]
        sub = df[(df["method"] == "lora") & (df["rank"] == v["r"]) &
                 (df["targets"] == v["targets"])] if not df.empty else df
        row.append(f"{int(sub['trainable_params'].iloc[0]) - int(sub['head_params'].iloc[0]):,}"
                   if len(sub) else "–")
        for t in tasks:
            s = stat(best_scores(df, method="lora", task=t, rank=v["r"], targets=v["targets"]))
            row += [pm(s) + (f" (n={s['n']})" if s and s["n"] != 5 else ""),
                    f1(s["median"]) if s else "–"]
        rows.append(row)
    h = ["Adapted matrices", "r", "LoRA params (excl. head)"]
    for t in tasks:
        h += [f"{NAME[t]}: mean ± std", f"{NAME[t]}: median"]
    return md_table(h, rows, ["l"] + ["r"] * (len(h) - 1))


def block_variance(df, paper) -> str:
    rows = []
    for t in SMALL_FIRST:
        lo = stat(best_scores(df, method="lora", task=t, rank=8, targets=CORE_TARGETS))
        ft = stat(best_scores(df, method="ft", task=t))
        pc = paper_cell(paper, "lora", t)
        rows.append([NAME[t], lo["n"] if lo else 0, f1(lo["std"]) if lo else "–",
                     f1(lo["max"] - lo["min"]) if lo else "–", f1(ft["std"]) if ft else "–",
                     f1(ft["max"] - ft["min"]) if ft else "–", f1(pc.get("std"))])
    h = ["Task", "Seeds", "LoRA std", "LoRA max − min", "FT std", "FT max − min",
         "Paper LoRA ± (as printed)"]
    return md_table(h, rows)


def block_variance_note(df) -> str:
    def mean_std(method, tasks):
        vals = []
        for t in tasks:
            kw = {"method": method, "task": t}
            if method == "lora":
                kw.update(rank=8, targets=CORE_TARGETS)
            s = stat(best_scores(df, **kw))
            if s and not np.isnan(s["std"]):
                vals.append(s["std"])
        return float(np.mean(vals)) if len(vals) == len(tasks) else None

    parts = []
    for method, label in (("lora", "LoRA"), ("ft", "full fine-tuning")):
        a, b = mean_std(method, ["rte", "mrpc"]), mean_std(method, ["sst2", "cola", "stsb"])
        if a is None or b is None:
            continue
        rel = "larger than" if a > b else "not larger than"
        parts.append(f"for {label} the mean seed std of RTE and MRPC ({a:.1f}) is {rel} that of "
                     f"SST-2, CoLA and STS-B ({b:.1f})")
    if not parts:
        return "_Not enough finished runs for the seed-variance comparison._"
    return ("Seed sensitivity, computed from the table above: " + "; ".join(parts) + ". "
            "The dev sets of RTE and MRPC are small, so a single example changes accuracy by a "
            "visible amount; the paper's own reported spread is printed in the last column.")


def block_selection(df) -> str:
    rows = []
    for t, name, _ in TASKS:
        kw = dict(method="lora", task=t, rank=8, targets=CORE_TARGETS)
        b, last = stat(best_scores(df, **kw)), stat(last_scores(df, **kw))
        rows.append([name, f1(b["mean"]) if b else "–", f1(last["mean"]) if last else "–",
                     signed(b["mean"] - last["mean"]) if b and last else "–"])
    return md_table(["Task", "Best-epoch mean", "Last-epoch mean", "Difference"], rows)


# ---------------------------------------------------------------- bench blocks
def block_bench() -> str:
    p = RESULTS / "bench.json"
    if not p.exists():
        return "_Benchmarks have not been run._"
    b = json.loads(p.read_text(encoding="utf-8"))
    out = [f"Measured on {b['gpu']} (torch {b['torch']}), commit `{b['commit']}`.", ""]
    rows = []
    for r in b["sizes"]["params"]:
        if r["method"] == "ft":
            rows.append(["Full fine-tuning", "–", f"{r['trainable']:,}", f"{r['trainable']:,}",
                         "100.0 %"])
        else:
            rows.append([f"LoRA r={r['rank']}", f"{r['lora']:,}", f"{r['head']:,}",
                         f"{r['trainable']:,}",
                         f"{100 * r['trainable'] / b['sizes']['params'][0]['total']:.2f} %"])
    out += ["**Trainable parameters** (RoBERTa-base with a two-class head; the head is "
            "randomly initialised and always trained, and is not part of the paper's count)", "",
            md_table(["Method", "LoRA A/B", "Head", "Trainable total", "Share of full model"],
                     rows), ""]
    f = b["sizes"]["files"]
    full = f["full_checkpoint_bytes"]
    rows = [["Full checkpoint (state dict, fp32)", f"{full / 2**20:.1f} MiB", "1×"]]
    for r in RANKS:
        for key, lab in ((f"adapter_r{r}_ab_only_bytes", "A and B only"),
                         (f"adapter_r{r}_ab_head_bytes", "A, B and head")):
            rows.append([f"LoRA r={r}, {lab}", f"{f[key] / 2**20:.2f} MiB",
                         f"{full / f[key]:.0f}× smaller"])
    out += ["**Adapter versus full checkpoint (file size)**", "",
            md_table(["Artifact", "Size", "Relative"], rows), ""]
    lat = [x for x in b["latency"] if "batch" in x]
    rows = []
    for bs, seq, prec in sorted({(x["batch"], x["seq"], x["precision"]) for x in lat}):
        u = next(x for x in lat if (x["batch"], x["seq"], x["precision"], x["state"]) ==
                 (bs, seq, prec, "unmerged"))
        m = next(x for x in lat if (x["batch"], x["seq"], x["precision"], x["state"]) ==
                 (bs, seq, prec, "merged"))
        rows.append([bs, seq, prec, f"{u['median_ms']:.2f}", f"{m['median_ms']:.2f}",
                     f"{u['median_ms'] / m['median_ms']:.2f}×"])
    diff = [x for x in b["latency"] if "max_abs_logit_diff_merged_vs_unmerged" in x]
    out += ["**Inference latency, merged versus unmerged** (median over many iterations after "
            "warm-up, synchronised timing; batch size 1 is launch-bound and noisy)", "",
            md_table(["Batch", "Seq. len.", "Precision", "Unmerged (ms)", "Merged (ms)",
                      "Speed-up"], rows)]
    if diff:
        out += ["", f"Maximum absolute logit difference, merged vs unmerged (fp32): "
                    f"{diff[0]['max_abs_logit_diff_merged_vs_unmerged']:.2e}."]
    rows = []
    for bs in sorted({x["batch"] for x in b["memory"]}):
        for mode in ("train", "inference"):
            lo = next(x for x in b["memory"] if (x["method"], x["mode"], x["batch"]) ==
                      ("lora", mode, bs))
            ft = next(x for x in b["memory"] if (x["method"], x["mode"], x["batch"]) ==
                      ("ft", mode, bs))
            rows.append([mode, bs, lo["seq"], f"{lo['peak_mib']:.0f}", f"{ft['peak_mib']:.0f}",
                         f"{100 * (1 - lo['peak_mib'] / ft['peak_mib']):.0f} %"])
    out += ["", "**Peak GPU memory** (bf16 autocast, fp32 parameters, AdamW for training; "
                "LoRA inference uses merged weights)", "",
            md_table(["Mode", "Batch", "Seq. len.", "LoRA (MiB)", "Full FT (MiB)",
                      "LoRA saving"], rows)]
    return "\n".join(out)


# ---------------------------------------------------------------- setup / coverage / runtime
def block_setup(df) -> str:
    lora = yaml.safe_load((ROOT / "configs" / "lora_roberta_base.yaml").read_text("utf-8"))
    ft = yaml.safe_load((ROOT / "configs" / "ft_roberta_base.yaml").read_text("utf-8"))
    rows = []
    for t, name, _ in TASKS:
        a, b = lora["tasks"][t], ft["tasks"][t]
        rows.append([name, f"{a['lr']:g}", a["batch_size"], a["epochs"],
                     f"{b['lr']:g}", b["batch_size"], b["epochs"]])
    tab = md_table(["Task", "LoRA lr", "LoRA batch", "LoRA epochs", "FT lr", "FT batch",
                    "FT epochs"], rows)
    seeds = sorted(df["seed"].unique()) if not df.empty else []
    lc, oc = lora["lora"], lora["optim"]
    gpu = df["gpu"].iloc[0] if not df.empty else "–"
    lines = [
        f"- Model: `{lora['model']}`; LoRA on {' and '.join(lc['targets'])} projections, "
        f"r = {lc['r']}, alpha = {lc['alpha']}, LoRA dropout {lc['dropout']}, classification "
        f"head trained.",
        f"- Optimiser: AdamW, linear schedule with {oc['warmup_ratio']:g} warm-up ratio, weight "
        f"decay {oc['weight_decay']:g}, betas {tuple(oc['betas'])}, epsilon {oc['eps']:g}, "
        f"gradient clipping {oc['max_grad_norm']:g}; maximum sequence length "
        f"{lora['max_seq_len']}; {lora['precision']} autocast.",
        f"- Seeds: {', '.join(str(s) for s in seeds)}." if seeds else "- Seeds: none finished.",
        f"- Hardware and libraries: {gpu}; torch {df['torch'].iloc[0] if not df.empty else '–'}, "
        f"transformers {df['transformers'].iloc[0] if not df.empty else '–'}.",
        "- Per-task values below: LoRA from Table 9 of the paper, FT from the fixed point of the "
        "RoBERTa fine-tuning grid ([`configs/lora_roberta_base.yaml`](configs/"
        "lora_roberta_base.yaml), [`configs/ft_roberta_base.yaml`](configs/ft_roberta_base.yaml)).",
    ]
    return "\n".join(lines) + "\n\n" + tab


def expected_keys() -> dict[str, list]:
    m = load_matrix()
    return {g: build(g, m) for g in ("core", "ablation", "proj_ablation")}


def run_key(r) -> tuple:
    if r["method"] == "ft":
        return ("ft", r["task"], "", "", "", str(r["seed"]))
    return ("lora", r["task"], str(int(r["rank"])), f"{float(r['alpha']):g}", r["targets"],
            str(r["seed"]))


def block_coverage(df) -> tuple[str, int]:
    done = set(run_key(r) for _, r in df.iterrows()) if not df.empty else set()
    rows = []
    unique = {e.key for g in expected_keys().values() for e in g}
    missing = len(unique - done)
    for g, label in (("core", "Core (LoRA and FT)"), ("ablation", "Rank ablation"),
                     ("proj_ablation", "Projection ablation")):
        ent = expected_keys()[g]
        for t in SMALL_FIRST:
            e = [x for x in ent if x.task == t]
            if not e:
                continue
            n_done = sum(1 for x in e if x.key in done)
            rows.append([label, NAME[t], len(e), n_done, len(e) - n_done])
    div = int((df["status"] == "diverged").sum()) if not df.empty else 0
    fail = RESULTS / "failures.csv"
    nfail = max(0, sum(1 for _ in open(fail, encoding="utf-8")) - 1) if fail.exists() else 0
    tab = md_table(["Group", "Task", "Planned", "Finished", "Not run"], rows)
    tab += f"\n\nRuns flagged as diverged (non-finite loss): {div}. Permanent failures: {nfail}."
    return tab, missing


def block_runtime(df) -> str:
    import estimate_compute as ec

    try:
        times = ec.probe_times()
    except FileNotFoundError:
        return "_No timing probe available._"
    m = load_matrix()
    ent = expected_keys()
    rows = []
    done = {run_key(r): r for _, r in df.iterrows()} if not df.empty else {}
    for g, label, hours in (
        ("core", "`make core`", sum(ec.core_hours(m, times).values())),
        ("ablation", "`make ablation` (ranks other than the default)",
         sum(ec.rank_hours(m, times).values())),
        ("proj_ablation", "`make proj_ablation` (variants other than the default)",
         sum(ec.proj_hours(m, times).values())),
    ):
        measured = [done[e.key]["train_time_s"] for e in ent[g] if e.key in done
                    and (g == "core" or not _is_core(e, ent["core"]))]
        n_all = len([e for e in ent[g] if g == "core" or not _is_core(e, ent["core"])])
        rows.append([label, f"{hours:.1f}", f"{sum(measured) / 3600:.1f}",
                     f"{len(measured)} / {n_all}"])
    return md_table(["Target", "Estimated GPU hours (probe)", "Measured train hours",
                     "Runs finished"], rows)


def _is_core(entry, core_entries) -> bool:
    return entry.key in {c.key for c in core_entries}


# ---------------------------------------------------------------- findings
def findings(df, info, paper) -> list[str]:
    out = []
    have = [t for t in info if info[t]["stat"]]
    if len(have) == len(TASKS):
        ms = np.mean([info[t]["stat"]["median"] for t in info])
        ps = np.mean([info[t]["paper"]["value"] for t in info])
        out.append(f"Across the five tasks, the mean of the per-task LoRA medians is {ms:.1f} "
                   f"against {ps:.1f} for the published LoRA row ({ms - ps:+.1f}).")
    if have:
        t = max(have, key=lambda k: abs(info[k]["delta"]))
        d = info[t]
        out.append(f"The largest per-task difference is on {NAME[t]}: median "
                   f"{d['stat']['median']:.1f} versus {d['paper']['value']:.1f} published "
                   f"({d['delta']:+.1f}). MRPC, RTE and STS-B start from the pre-trained model "
                   f"here and from an MNLI-adapted checkpoint in the published row.")
    wins, comp, gaps = 0, 0, []
    for t, _, _ in TASKS:
        lo = stat(best_scores(df, method="lora", task=t, rank=8, targets=CORE_TARGETS))
        ft = stat(best_scores(df, method="ft", task=t))
        if lo and ft:
            comp += 1
            wins += lo["median"] > ft["median"]
            gaps.append(lo["median"] - ft["median"])
    if comp:
        out.append(f"LoRA (median) is above my full fine-tuning run on {wins} of {comp} tasks; "
                   f"the mean difference of medians is {np.mean(gaps):+.1f}. The FT run uses one "
                   f"fixed setting and no MNLI start.")
    close = tot = 0
    for t, _, _ in TASKS:
        a = stat(best_scores(df, method="lora", task=t, rank=1, targets=CORE_TARGETS))
        b = stat(best_scores(df, method="lora", task=t, rank=8, targets=CORE_TARGETS))
        if a and b and not np.isnan(a["std"]):
            tot += 1
            close += abs(a["mean"] - b["mean"]) <= max(a["std"], b["std"])
    if tot:
        out.append(f"Rank ablation: the mean score at r = 1 differs from r = 8 by no more than "
                   f"the larger seed std on {close} of {tot} tasks.")
    stds = {t: stat(best_scores(df, method="lora", task=t, rank=8,
                                targets=CORE_TARGETS)) for t, _, _ in TASKS}
    stds = {t: s["std"] for t, s in stds.items() if s and not np.isnan(s["std"])}
    if len(stds) == len(TASKS):
        hi = max(stds, key=stds.get)
        lo = min(stds, key=stds.get)
        small = np.mean([stds["rte"], stds["mrpc"]])
        big = np.mean([stds["sst2"], stds["cola"], stds["stsb"]])
        out.append(f"Seed sensitivity: the LoRA std is largest on {NAME[hi]} ({stds[hi]:.1f}) and "
                   f"smallest on {NAME[lo]} ({stds[lo]:.1f}); the mean std of RTE and MRPC is "
                   f"{small:.1f} against {big:.1f} for SST-2, CoLA and STS-B.")
    return ["- " + s for s in out[:5]]


# ---------------------------------------------------------------- figures
def _axes_strip(ax, groups, colors, labels):
    for i, (vals, c) in enumerate(zip(groups, colors, strict=True)):
        if not len(vals):
            continue
        x = i + np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([i])
        ax.boxplot([vals], positions=[i], widths=0.5, showfliers=False,
                   medianprops={"color": c, "linewidth": 1.6},
                   boxprops={"color": c, "linewidth": 1.0},
                   whiskerprops={"color": c, "linewidth": 1.0},
                   capprops={"color": c, "linewidth": 1.0})
        ax.scatter(x, vals, s=22, color=c, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def fig_variance(df):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
    for ax, t in zip(axes, SMALL_FIRST, strict=True):
        lo = best_scores(df, method="lora", task=t, rank=8, targets=CORE_TARGETS)
        ft = best_scores(df, method="ft", task=t)
        _axes_strip(ax, [lo, ft], [C1, C2], ["LoRA", "Full FT"])
        ax.set_title(NAME[t])
        ax.set_ylabel(METRIC[t] + " (×100)" if t == SMALL_FIRST[0] else "")
    fig.suptitle("Per-seed dev score at the best epoch (each point is one seed)", x=0.01,
                 ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "seed_variance.png", dpi=160)
    plt.close(fig)


def fig_rank(df, paper):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
    for ax, t in zip(axes, SMALL_FIRST, strict=True):
        means, stds, xs = [], [], []
        for i, r in enumerate(RANKS):
            v = best_scores(df, method="lora", task=t, rank=r, targets=CORE_TARGETS)
            if not v:
                continue
            ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=14,
                       color=C1, alpha=0.45, edgecolor="none", zorder=2)
            s = stat(v)
            xs.append(i)
            means.append(s["mean"])
            stds.append(0 if np.isnan(s["std"]) else s["std"])
        if xs:
            ax.errorbar(xs, means, yerr=stds, color=C1, marker="o", markersize=5, linewidth=1.5,
                        capsize=3, zorder=3, label="mine, mean ± std")
        ax.axhline(paper_cell(paper, "lora", t)["value"], color=C2, linestyle="--",
                   linewidth=1.2, label="paper, r = 8")
        ax.set_xticks(range(len(RANKS)), [str(r) for r in RANKS])
        ax.set_xlabel("rank r")
        ax.set_title(NAME[t])
        if t == SMALL_FIRST[0]:
            ax.set_ylabel("dev score (×100)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Rank ablation, alpha = r (points are single seeds)", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "rank_ablation.png", dpi=160)
    plt.close(fig)


def fig_proj(df):
    if df.empty:
        return
    tasks = ["rte", "mrpc", "stsb"]
    cols = [C1, C2, C3, C4]
    var = proj_variants()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, t in zip(axes, tasks, strict=True):
        for i, v in enumerate(var):
            vals = best_scores(df, method="lora", task=t, rank=v["r"], targets=v["targets"])
            if not vals:
                continue
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.1, 0.1, len(vals)), vals, s=18,
                       color=cols[i], alpha=0.55, edgecolor="none", zorder=2)
            s = stat(vals)
            ax.errorbar([i], [s["mean"]], yerr=[0 if np.isnan(s["std"]) else s["std"]],
                        color=cols[i], marker="D", markersize=5, capsize=3, linewidth=1.5,
                        zorder=3)
        ax.set_xticks(range(len(var)), [f"{v['label']}\nr={v['r']}" for v in var], fontsize=7)
        ax.set_title(NAME[t])
        if t == tasks[0]:
            ax.set_ylabel("dev score (×100)")
    fig.suptitle("Projection ablation at equal LoRA parameter count (diamond: mean ± std)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "projection_ablation.png", dpi=160)
    plt.close(fig)


def fig_curves(df):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0))
    for ax, t in zip(axes, SMALL_FIRST, strict=True):
        for method, color, label, kw in (("lora", C1, "LoRA", {"rank": 8, "targets": CORE_TARGETS}),
                                         ("ft", C2, "Full FT", {})):
            sub = pick(df, method=method, task=t, **kw)
            if sub.empty:
                continue
            n = min(len(c) for c in sub["curve"])
            arr = np.array([c[:n] for c in sub["curve"]])
            x = np.arange(1, n + 1)
            ax.plot(x, arr.mean(0), color=color, linewidth=1.6, label=label)
            ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.15, linewidth=0)
        ax.set_title(NAME[t])
        ax.set_xlabel("epoch")
        if t == SMALL_FIRST[0]:
            ax.set_ylabel("dev score (×100)")
            ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Dev score per epoch (line: mean over seeds, band: min to max)", x=0.01,
                 ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- README injection
def inject(readme: str, name: str, content: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if not pat.search(readme):
        raise KeyError(f"README is missing the block {name}")
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", readme)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    paper = yaml.safe_load((ROOT / "paper" / "paper_results.yaml").read_text("utf-8"))
    reasons = yaml.safe_load((ROOT / "docs" / "reasons.yaml").read_text("utf-8"))
    df = load_runs()

    top, info = block_paper_vs_mine(df, paper, reasons)
    rank_mean, rank_med = block_rank(df)
    cov, missing = block_coverage(df)
    blocks = {
        "paper_vs_mine": top,
        "key_findings": "\n".join(findings(df, info, paper)) or "_No finished runs yet._",
        "setup": block_setup(df),
        "core": block_core(df),
        "ft_comparison": block_ft(df, paper),
        "selection": block_selection(df),
        "rank_mean": rank_mean,
        "rank_median": rank_med,
        "proj": block_proj(df),
        "variance": block_variance(df, paper),
        "variance_note": block_variance_note(df),
        "bench": block_bench(),
        "coverage": cov,
        "runtime": block_runtime(df),
    }
    for name, text in blocks.items():
        (TABLES / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    fig_variance(df)
    fig_rank(df, paper)
    fig_proj(df)
    fig_curves(df)

    readme_path = ROOT / "README.md"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        for name, content in blocks.items():
            text = inject(text, name, content)
        readme_path.write_text(text, encoding="utf-8")
    print(f"report written; {len(df)} finished runs; {missing} planned runs not finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
