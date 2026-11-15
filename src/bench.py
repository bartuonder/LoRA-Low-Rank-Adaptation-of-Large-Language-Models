"""Engineering benchmarks: parameters, adapter size, merged vs unmerged latency, peak VRAM.

Run only while no training job is active (the script refuses to start otherwise), so that
timings and memory numbers are clean. Results go to results/bench.json.
"""
import argparse
import gc
import io
import json
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from lora.config import ROOT, git_commit
from lora.inject import (
    adapter_state_dict,
    count_parameters,
    inject_lora,
    lora_modules,
    merge_lora,
    unmerge_lora,
)

MODEL = "roberta-base"
DEV = torch.device("cuda")


def training_active() -> list[str]:
    me = psutil.Process().pid
    hits = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if p.info["pid"] != me and ("train.py" in cl or "run_queue.py" in cl
                                    or "timing_probe.py" in cl):
            hits.append(cl)
    return hits


def fresh(method: str, r: int = 8, targets=("query", "value")):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, attn_implementation="sdpa")
    if method == "lora":
        inject_lora(model, targets=targets, r=r, alpha=r)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():  # non-zero B so that merging is a real operation
            for m in lora_modules(model):
                m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.02)
    return model.to(DEV)


def sync_time(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    q = statistics.quantiles(times, n=4)
    return {"median_ms": statistics.median(times), "q1_ms": q[0], "q3_ms": q[2],
            "mean_ms": statistics.fmean(times), "iters": iters}


def make_forward(model, ids, mask, prec: str):
    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
            model(input_ids=ids, attention_mask=mask)

    return fwd


def bench_latency(iters: int, warmup: int) -> list[dict]:
    rows = []
    model = fresh("lora").eval()
    for bs, seq in [(1, 128), (16, 128), (32, 128)]:
        ids = torch.randint(5, 50000, (bs, seq), device=DEV)
        mask = torch.ones_like(ids)
        for prec in ("fp32", "bf16"):
            fwd = make_forward(model, ids, mask, prec)
            for state in ("unmerged", "merged"):
                (merge_lora if state == "merged" else unmerge_lora)(model)
                r = sync_time(fwd, warmup, iters)
                rows.append({"batch": bs, "seq": seq, "precision": prec, "state": state, **r})
                print(rows[-1], flush=True)
        unmerge_lora(model)
    # merged and unmerged logits agree (sanity, fp32)
    ids = torch.randint(5, 50000, (4, 64), device=DEV)
    with torch.no_grad():
        a = model(input_ids=ids).logits
        merge_lora(model)
        b = model(input_ids=ids).logits
        unmerge_lora(model)
    rows.append({"max_abs_logit_diff_merged_vs_unmerged": float((a - b).abs().max())})
    del model
    return rows


def train_step_peak(method: str, bs: int, seq: int) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    model = fresh(method)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, fused=True)
    ids = torch.randint(5, 50000, (bs, seq), device=DEV)
    mask = torch.ones_like(ids)
    y = torch.randint(0, 2, (bs,), device=DEV)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        F.cross_entropy(logits.float(), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, opt
    return {"method": method, "mode": "train", "batch": bs, "seq": seq, "peak_mib": peak}


def infer_peak(method: str, bs: int, seq: int) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    model = fresh(method).eval()
    if method == "lora":
        merge_lora(model)
    ids = torch.randint(5, 50000, (bs, seq), device=DEV)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model
    return {"method": method, "mode": "inference", "batch": bs, "seq": seq, "peak_mib": peak}


def nbytes_saved(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getbuffer().nbytes


def bench_sizes() -> dict:
    out = {"params": [], "files": {}}
    full = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    c = count_parameters(full)
    out["params"].append({"method": "ft", "lora": 0, "head": 0, "trainable": c["total"],
                          "total": c["total"]})
    out["files"]["full_checkpoint_bytes"] = nbytes_saved(full.state_dict())
    for r in (1, 2, 4, 8):
        m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        inject_lora(m, r=r, alpha=r)
        c = count_parameters(m)
        out["params"].append({"method": "lora", "rank": r, **c})
        sd = adapter_state_dict(m)
        ab_only = {k: v for k, v in sd.items() if "lora_" in k}
        out["files"][f"adapter_r{r}_ab_head_bytes"] = nbytes_saved(sd)
        out["files"][f"adapter_r{r}_ab_only_bytes"] = nbytes_saved(ab_only)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results" / "bench.json"))
    ap.add_argument("--force", action="store_true", help="skip the active-training check")
    args = ap.parse_args()
    if not args.force and (busy := training_active()):
        print("training job active, refusing to benchmark:\n  " + "\n  ".join(busy))
        return 1
    if not torch.cuda.is_available():
        print("CUDA required for the benchmarks")
        return 1

    result = {"commit": git_commit(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "model": MODEL}
    result["sizes"] = bench_sizes()
    result["latency"] = bench_latency(args.iters, args.warmup)
    result["memory"] = []
    for bs, seq in [(16, 128), (32, 128)]:
        for method in ("lora", "ft"):
            result["memory"].append(train_step_peak(method, bs, seq))
            print(result["memory"][-1], flush=True)
            result["memory"].append(infer_peak(method, bs, seq))
            print(result["memory"][-1], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
