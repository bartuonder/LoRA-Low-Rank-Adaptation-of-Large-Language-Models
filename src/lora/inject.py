import re

import torch.nn as nn

from .layer import LoRALinear

# Attention projections of an HF RoBERTa/BERT encoder layer.
# Wo is attention.output.dense; the FFN output projection (layer.output.dense) is not matched.
PROJECTION_PATTERNS = {
    "query": r"attention\.self\.query$",
    "key": r"attention\.self\.key$",
    "value": r"attention\.self\.value$",
    "output": r"attention\.output\.dense$",
}
ALIASES = {"q": "query", "k": "key", "v": "value", "o": "output", "wq": "query", "wk": "key",
           "wv": "value", "wo": "output"}
HEAD_PREFIX = "classifier."


def canonical_targets(targets) -> list[str]:
    out = []
    for t in targets:
        t = ALIASES.get(str(t).lower(), str(t).lower())
        if t not in PROJECTION_PATTERNS:
            raise ValueError(f"unknown projection {t!r}; choose from {sorted(PROJECTION_PATTERNS)}")
        if t not in out:
            out.append(t)
    return out


def inject_lora(
    model: nn.Module,
    targets=("query", "value"),
    r: int = 8,
    alpha: float = 8,
    dropout: float = 0.0,
    train_head: bool = True,
) -> list[str]:
    """Wrap the selected attention projections with LoRALinear and freeze everything else.

    Returns the names of the wrapped modules. Only LoRA A/B (and the classification head,
    when train_head is set) keep requires_grad=True.
    """
    targets = canonical_targets(targets)
    regexes = [re.compile(PROJECTION_PATTERNS[t]) for t in targets]
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(rx.search(name) for rx in regexes):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no modules matched targets {targets}")

    if train_head:
        for name, p in model.named_parameters():
            if name.startswith(HEAD_PREFIX):
                p.requires_grad_(True)
    return wrapped


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def merge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.unmerge()


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable parameter counts, with the LoRA part and the head reported separately."""
    lora = head = other = total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora += p.numel()
        elif name.startswith(HEAD_PREFIX):
            head += p.numel()
        else:
            other += p.numel()
    return {"lora": lora, "head": head, "other": other,
            "trainable": lora + head + other, "total": total}


def adapter_state_dict(model: nn.Module) -> dict:
    """A, B and head parameters only (what has to be stored per task)."""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n or n.startswith(HEAD_PREFIX)}
