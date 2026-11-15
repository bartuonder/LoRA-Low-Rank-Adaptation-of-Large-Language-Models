from dataclasses import dataclass

import numpy as np
import torch

TEXT_KEYS = {
    "sst2": ("sentence", None),
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
}
NUM_LABELS = {"stsb": 1, "mnli": 3}  # everything else is binary


def num_labels(task: str) -> int:
    return NUM_LABELS.get(task, 2)


@dataclass
class Split:
    input_ids: list[list[int]]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def load_glue(task: str, tokenizer, max_len: int, limit: int | None = None):
    """Tokenise the train and validation splits once (no padding; padding is dynamic)."""
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", task)
    k1, k2 = TEXT_KEYS[task]
    out = {}
    for name in ("train", "validation"):
        d = ds[name]
        if limit:
            d = d.select(range(min(limit, len(d))))
        a = list(d[k1])  # datasets>=5 returns lazy columns
        b = list(d[k2]) if k2 else None
        enc = tokenizer(a, b, truncation=True, max_length=max_len)
        dtype = np.float32 if task == "stsb" else np.int64
        out[name] = Split(enc["input_ids"], np.asarray(list(d["label"]), dtype=dtype))
    return out["train"], out["validation"]


def collate(split: Split, idx, pad_id: int, regression: bool, device=None):
    ids = [split.input_ids[i] for i in idx]
    n = max(len(x) for x in ids)
    inp = torch.full((len(ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for j, x in enumerate(ids):
        inp[j, : len(x)] = torch.tensor(x)
        mask[j, : len(x)] = 1
    labels = torch.from_numpy(split.labels[np.asarray(idx)])
    batch = (inp, mask, labels)
    if device is not None:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
    return batch


def train_batches(split: Split, batch_size: int, seed: int, epoch: int):
    """Shuffled index batches; the order depends only on (seed, epoch), so resume is exact."""
    g = np.random.default_rng([seed, epoch])
    perm = g.permutation(len(split))
    for i in range(0, len(perm), batch_size):
        yield perm[i : i + batch_size]


def eval_batches(split: Split, batch_size: int):
    """Length-sorted batches for fast evaluation; labels are gathered by index."""
    order = np.argsort([len(x) for x in split.input_ids], kind="stable")
    for i in range(0, len(order), batch_size):
        yield order[i : i + batch_size]
