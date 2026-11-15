import copy

import pytest
import torch

peft = pytest.importorskip("peft")

from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

from lora.inject import count_parameters, inject_lora, lora_modules  # noqa: E402
from lora.layer import LoRALinear  # noqa: E402

CASES = {
    "qv": (("query", "value"), ["query", "value"]),
    "qkvo": (("query", "key", "value", "output"),
             r".*attention\.(self\.(query|key|value)|output\.dense)"),
}


def copy_into_peft(mine, peft_model) -> int:
    """Copy my A/B into the PEFT LoRA layers (same module path), return how many were set."""
    peft_layers = {n.split("base_model.model.")[-1]: m for n, m in peft_model.named_modules()
                   if hasattr(m, "lora_A") and hasattr(m, "lora_B")}
    n = 0
    for name, m in mine.named_modules():
        if isinstance(m, LoRALinear):
            layer = peft_layers[name]
            with torch.no_grad():
                layer.lora_A["default"].weight.copy_(m.lora_A)
                layer.lora_B["default"].weight.copy_(m.lora_B)
            n += 1
    return n


@pytest.mark.parametrize("case", list(CASES))
def test_logits_match_peft(tiny, batch, case):
    targets, peft_targets = CASES[case]
    r, alpha = 4, 8
    base_for_peft = copy.deepcopy(tiny)
    inject_lora(tiny, targets=targets, r=r, alpha=alpha)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for m in lora_modules(tiny):
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.05)

    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=0.0,
                     target_modules=peft_targets, bias="none")
    pm = get_peft_model(base_for_peft, cfg).eval()
    assert copy_into_peft(tiny, pm) == len(lora_modules(tiny))

    ids, mask = batch
    with torch.no_grad():
        mine = tiny(input_ids=ids, attention_mask=mask).logits
        theirs = pm(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(mine, theirs, atol=1e-5, rtol=1e-5)


def test_trainable_counts_match_peft(tiny):
    base = copy.deepcopy(tiny)
    inject_lora(tiny, targets=("query", "value"), r=4, alpha=4, train_head=True)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=4, lora_alpha=4,
                     target_modules=["query", "value"])
    pm = get_peft_model(base, cfg)
    peft_lora = sum(p.numel() for n, p in pm.named_parameters()
                    if p.requires_grad and "lora_" in n)
    # PEFT trains the classifier through a duplicated modules_to_save copy
    peft_head = sum(p.numel() for n, p in pm.named_parameters()
                    if p.requires_grad and "modules_to_save" in n)
    mine = count_parameters(tiny)
    assert mine["lora"] == peft_lora
    assert mine["head"] == peft_head
