
import pytest
import torch

from lora.inject import (
    adapter_state_dict,
    count_parameters,
    inject_lora,
    lora_modules,
    merge_lora,
    unmerge_lora,
)
from lora.layer import LoRALinear

TARGET_SETS = [("query",), ("value",), ("query", "value"), ("query", "key", "value", "output")]


def randomize_b(model, scale: float = 0.05, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in lora_modules(model):
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * scale)


def logits(model, batch):
    ids, mask = batch
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits


@pytest.mark.parametrize("targets", TARGET_SETS)
def test_zero_init_matches_base(tiny, tiny_copy, batch, targets):
    inject_lora(tiny, targets=targets, r=4, alpha=4)
    assert torch.equal(logits(tiny, batch), logits(tiny_copy, batch))


def test_scaling_and_shapes(tiny):
    inject_lora(tiny, targets=("query", "value"), r=4, alpha=8)
    mods = lora_modules(tiny)
    assert len(mods) == 2 * tiny.config.num_hidden_layers
    for m in mods:
        assert m.scaling == 2.0
        assert m.lora_A.shape == (4, 64) and m.lora_B.shape == (64, 4)
        assert torch.count_nonzero(m.lora_B) == 0
        assert m.lora_A.std() > 0


@pytest.mark.parametrize("targets", TARGET_SETS)
def test_only_expected_params_trainable(tiny, targets):
    inject_lora(tiny, targets=targets, r=2, alpha=2, train_head=True)
    for name, p in tiny.named_parameters():
        expected = "lora_A" in name or "lora_B" in name or name.startswith("classifier.")
        assert p.requires_grad == expected, name
    n_proj = len(targets) * tiny.config.num_hidden_layers
    c = count_parameters(tiny)
    assert c["lora"] == n_proj * 2 * 64 * 2  # r * (in + out) per matrix
    assert c["other"] == 0
    assert c["trainable"] == c["lora"] + c["head"]


def test_head_can_be_frozen(tiny):
    inject_lora(tiny, r=2, alpha=2, train_head=False)
    assert count_parameters(tiny)["head"] == 0


def test_ffn_output_is_not_wrapped(tiny):
    inject_lora(tiny, targets=("output",), r=2, alpha=2)
    wrapped = {n for n, m in tiny.named_modules() if isinstance(m, LoRALinear)}
    assert all(n.endswith("attention.output.dense") for n in wrapped) and wrapped


def test_unknown_target_raises(tiny):
    with pytest.raises(ValueError):
        inject_lora(tiny, targets=("mlp",))


def test_merge_unmerge_roundtrip(tiny, batch):
    inject_lora(tiny, targets=("query", "key", "value", "output"), r=4, alpha=8)
    randomize_b(tiny)
    base_w = {n: p.detach().clone() for n, p in tiny.named_parameters() if ".base.weight" in n}
    before = logits(tiny, batch)
    merge_lora(tiny)
    assert all(m.merged for m in lora_modules(tiny))
    merged = logits(tiny, batch)
    assert torch.allclose(before, merged, atol=1e-5, rtol=1e-5)
    unmerge_lora(tiny)
    after = logits(tiny, batch)
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
    for n, p in tiny.named_parameters():
        if ".base.weight" in n:
            assert torch.allclose(p, base_w[n], atol=1e-6)


def test_lora_changes_output_after_training_signal(tiny, tiny_copy, batch):
    inject_lora(tiny, r=4, alpha=4)
    randomize_b(tiny)
    assert not torch.allclose(logits(tiny, batch), logits(tiny_copy, batch), atol=1e-4)


def test_gradients_reach_a_and_b_only(tiny, batch):
    inject_lora(tiny, r=4, alpha=4)
    randomize_b(tiny)
    tiny.train()
    ids, mask = batch
    tiny(input_ids=ids, attention_mask=mask).logits.sum().backward()
    for n, p in tiny.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        else:
            assert p.grad is None, n


def test_adapter_state_dict_contents(tiny):
    inject_lora(tiny, r=2, alpha=2)
    names = adapter_state_dict(tiny).keys()
    assert all("lora_" in n or n.startswith("classifier.") for n in names)
    assert len(names) == 2 * 2 * 2 + 4  # (A, B) x (q, v) x 2 layers + dense/out_proj weight+bias


@pytest.mark.slow
def test_roberta_base_trainable_count():
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    inject_lora(model, targets=("query", "value"), r=8, alpha=8)
    c = count_parameters(model)
    assert c["lora"] == 294_912  # 12 layers x 2 matrices x 2 x 768 x 8 (paper: 0.3M)
    assert c["head"] == 768 * 768 + 768 + 768 * 2 + 2
