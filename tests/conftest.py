import copy

import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification


def make_tiny_model(num_labels: int = 2, seed: int = 0) -> RobertaForSequenceClassification:
    torch.manual_seed(seed)
    cfg = RobertaConfig(
        vocab_size=120, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=40, num_labels=num_labels,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    return RobertaForSequenceClassification(cfg).eval()


@pytest.fixture
def tiny():
    return make_tiny_model()


@pytest.fixture
def tiny_copy(tiny):
    return copy.deepcopy(tiny)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 120, (4, 12), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0
    return ids, mask
