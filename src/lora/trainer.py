import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import config_hash, git_commit, run_id
from .data import collate, eval_batches, load_glue, num_labels, train_batches
from .inject import adapter_state_dict, count_parameters, inject_lora
from .metrics import compute_metric

RUNS_COLUMNS = [
    "run_id", "task", "method", "rank", "alpha", "targets", "seed", "epochs", "lr", "batch_size",
    "metric", "status", "best_epoch", "best_metric", "final_epoch", "final_metric",
    "epoch_metrics", "trainable_params", "head_params", "train_time_s", "config_hash",
    "commit", "torch", "transformers", "gpu", "finished_at",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, n_labels: int):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=n_labels, attn_implementation="sdpa"
    )
    if cfg["method"] == "lora":
        lc = cfg["lora"]
        inject_lora(model, targets=lc["targets"], r=lc["r"], alpha=lc["alpha"],
                    dropout=lc["dropout"], train_head=lc["train_head"])
    return model


def make_optimizer(model, cfg: dict):
    oc = cfg["optim"]
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith(".bias") or "LayerNorm" in n) else decay).append(p)
    groups = [{"params": decay, "weight_decay": oc["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(oc["betas"]), eps=oc["eps"],
                             fused=torch.cuda.is_available())


def make_scheduler(opt, total_steps: int, warmup_ratio: float):
    warm = int(total_steps * warmup_ratio)

    def f(step: int) -> float:
        if step < warm:
            return step / max(1, warm)
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def evaluate(model, split, cfg: dict, device, pad_id: int) -> float:
    model.eval()
    regression = cfg["task"] == "stsb"
    preds = np.zeros(len(split), dtype=np.float32 if regression else np.int64)
    for idx in eval_batches(split, 128):
        inp, mask, _ = collate(split, idx, pad_id, regression, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16"):
            logits = model(input_ids=inp, attention_mask=mask).logits
        logits = logits.float()
        preds[idx] = (logits.squeeze(-1) if regression else logits.argmax(-1)).cpu().numpy()
    return compute_metric(cfg["metric"], preds, split.labels)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_state(path: Path, model, opt, sched, epoch: int, history: list, meta: dict,
               best: dict | None = None) -> None:
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    state = {"trainable": trainable, "opt": opt.state_dict(), "sched": sched.state_dict(),
             "rng": _rng_state(), "epoch": epoch, "history": history, "meta": meta,
             "best": best}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state(path: Path, model, opt, sched, meta: dict):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["meta"] != meta:
        return None
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state["trainable"].items():
            params[n].copy_(t)
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    _set_rng_state(state["rng"])
    return state["epoch"], state["history"], state.get("best")


def save_best_adapter(adapters_dir: Path, rid: str, task: str, seed: int, best: dict,
                      cfg: dict) -> Path:
    """Best-epoch LoRA A/B and head weights, at adapters/<task>/seed<k>/<run_id>.pt.

    The run id is part of the file name because rank and projection ablations share
    (task, seed) with the core runs.
    """
    d = adapters_dir / task / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rid}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": best["state"], "best_epoch": best["epoch"],
                "best_metric": best["metric"], "metric_name": cfg["metric"],
                "lora": cfg["lora"], "task": task, "seed": seed}, tmp)
    os.replace(tmp, path)
    return path


def append_run_row(results_dir: Path, row: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "runs.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUNS_COLUMNS})


def env_info() -> dict:
    import datasets
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
    }


def run_training(cfg: dict, seed: int, results_dir: Path, ckpt_dir: Path,
                 data_limit: int | None = None, log=None,
                 adapters_dir: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    log = log or (lambda *a: print(*a, flush=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rid = run_id(cfg, seed)
    chash = config_hash(cfg)
    task = cfg["task"]
    regression = task == "stsb"
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    train, dev = load_glue(task, tok, cfg["max_seq_len"], limit=data_limit)
    model = build_model(cfg, num_labels(task)).to(device)
    counts = count_parameters(model)
    opt = make_optimizer(model, cfg)

    bs = cfg["batch_size"]
    micro = cfg.get("micro_batch_size") or bs
    steps_per_epoch = math.ceil(len(train) / bs)
    epochs = cfg["epochs"]
    sched = make_scheduler(opt, epochs * steps_per_epoch, cfg["optim"]["warmup_ratio"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{rid}.pt"
    meta = {"config_hash": chash, "seed": seed, "data_limit": data_limit}
    history, start, best = [], 0, None
    if ckpt.exists():
        loaded = load_state(ckpt, model, opt, sched, meta)
        if loaded is None:
            ckpt.unlink()
        else:
            start, history, best = loaded
            log(f"[{rid}] resumed after epoch {start}")
    keep_best = cfg["method"] == "lora" and adapters_dir is not None

    status = "ok"
    t_train = sum(h["epoch_time_s"] for h in history)
    log(f"[{rid}] trainable={counts['trainable']:,} (lora={counts['lora']:,} "
        f"head={counts['head']:,}) steps/epoch={steps_per_epoch}")
    for epoch in range(start, epochs):
        model.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        n_steps = 0
        for idx in train_batches(train, bs, seed, epoch):
            for sub in np.array_split(idx, math.ceil(len(idx) / micro)):
                inp, mask, y = collate(train, sub, tok.pad_token_id, regression, device)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=cfg["precision"] == "bf16"):
                    logits = model(input_ids=inp, attention_mask=mask).logits
                logits = logits.float()
                if regression:
                    loss = F.mse_loss(logits.squeeze(-1), y.float())
                else:
                    loss = F.cross_entropy(logits, y)
                (loss * len(sub) / len(idx)).backward()
                loss_sum += loss.detach() * len(sub) / len(idx)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["optim"]["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            n_steps += 1
        train_loss = loss_sum.item() / max(1, n_steps)
        epoch_time = time.time() - t0
        score = evaluate(model, dev, cfg, device, tok.pad_token_id)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "dev_metric": score,
                        "lr_end": sched.get_last_lr()[0], "epoch_time_s": epoch_time})
        t_train += epoch_time
        log(f"[{rid}] epoch {epoch + 1}/{epochs} loss={train_loss:.4f} "
            f"{cfg['metric']}={score:.2f} ({epoch_time:.0f}s)")
        if keep_best and math.isfinite(train_loss) and (best is None or score > best["metric"]):
            best = {"epoch": epoch + 1, "metric": score, "state": adapter_state_dict(model)}
        if not math.isfinite(train_loss):
            status = "diverged"
            break
        save_state(ckpt, model, opt, sched, epoch + 1, history, meta, best)

    scores = [h["dev_metric"] for h in history]
    best_i = int(np.argmax(scores))  # first maximum
    row = {
        "run_id": rid, "task": task, "method": cfg["method"],
        "rank": cfg["lora"]["r"] if cfg["method"] == "lora" else "",
        "alpha": cfg["lora"]["alpha"] if cfg["method"] == "lora" else "",
        "targets": "+".join(cfg["lora"]["targets"]) if cfg["method"] == "lora" else "",
        "seed": seed, "epochs": epochs, "lr": cfg["lr"], "batch_size": bs,
        "metric": cfg["metric"], "status": status,
        "best_epoch": best_i + 1, "best_metric": scores[best_i],
        "final_epoch": len(scores), "final_metric": scores[-1],
        "epoch_metrics": json.dumps([round(s, 4) for s in scores]),
        "trainable_params": counts["trainable"], "head_params": counts["head"],
        "train_time_s": round(t_train, 1), "config_hash": chash, "commit": git_commit(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    env = env_info()
    row.update({"torch": env["torch"], "transformers": env["transformers"], "gpu": env["gpu"]})
    detail = results_dir / "runs"
    detail.mkdir(parents=True, exist_ok=True)
    (detail / f"{rid}.json").write_text(
        json.dumps({"row": row, "config": cfg, "env": env, "history": history}, indent=1),
        encoding="utf-8")
    if keep_best and best is not None:
        save_best_adapter(adapters_dir, rid, task, seed, best, cfg)
    append_run_row(results_dir, row)
    if ckpt.exists():
        ckpt.unlink()
    return row
