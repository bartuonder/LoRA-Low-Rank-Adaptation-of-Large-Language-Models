import json

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification

from lora import trainer
from lora.config import ROOT, config_hash, load_config, run_id
from lora.data import Split, train_batches
from lora.metrics import compute_metric
from lora.trainer import run_training

LORA_CFG = ROOT / "configs" / "lora_roberta_base.yaml"
FT_CFG = ROOT / "configs" / "ft_roberta_base.yaml"


def test_metrics():
    y = np.array([0, 1, 1, 0])
    assert compute_metric("accuracy", np.array([0, 1, 0, 0]), y) == 75.0
    assert compute_metric("matthews_correlation", y, y) == 100.0
    assert compute_metric("pearson", np.array([1.0, 2, 3, 4]), np.array([2.0, 4, 6, 8])) == \
        pytest.approx(100.0)
    assert compute_metric("pearson", np.ones(4), np.arange(4.0)) == 0.0


def test_config_overrides_and_scope():
    cfg = load_config(LORA_CFG, "rte", ["lora.r=4", "lora.alpha=4", "epochs=2",
                                         "lora.targets=[query]"])
    assert cfg["lora"]["r"] == 4 and cfg["epochs"] == 2 and cfg["lora"]["targets"] == ["query"]
    assert cfg["lr"] == 5.0e-4 and cfg["batch_size"] == 32 and cfg["epochs"] == 2
    assert load_config(LORA_CFG, "rte", ["lr=1e-4"])["lr"] == 1e-4  # float, not str
    assert run_id(cfg, 3) == "lora_rte_r4_q_s3"
    assert config_hash(cfg) != config_hash(load_config(LORA_CFG, "rte"))
    with pytest.raises(ValueError):
        load_config(LORA_CFG, "mnli")  # out of scope


def test_paper_hyperparameters_in_config():
    import yaml

    paper = yaml.safe_load((ROOT / "paper" / "paper_results.yaml").read_text(encoding="utf-8"))
    hp = paper["hyperparameters_table9_base"]
    for task, key in [("sst2", "SST-2"), ("mrpc", "MRPC"), ("cola", "CoLA"), ("rte", "RTE"),
                      ("stsb", "STS-B")]:
        cfg = load_config(LORA_CFG, task)
        exp = hp["per_task"][key]
        assert (cfg["batch_size"], cfg["epochs"], cfg["lr"]) == (
            exp["batch_size"], exp["epochs"], exp["lr"])
        assert cfg["lora"]["r"] == hp["lora_rank"]["q"] and cfg["lora"]["alpha"] == hp["lora_alpha"]
        assert cfg["max_seq_len"] == hp["max_seq_len"]
        assert cfg["optim"]["warmup_ratio"] == hp["warmup_ratio"]


def test_batch_order_depends_only_on_seed_and_epoch():
    split = Split([[1]] * 50, np.zeros(50, dtype=np.int64))
    a = [b.tolist() for b in train_batches(split, 8, seed=1, epoch=2)]
    b = [b.tolist() for b in train_batches(split, 8, seed=1, epoch=2)]
    c = [b.tolist() for b in train_batches(split, 8, seed=1, epoch=3)]
    assert a == b and a != c
    assert sorted(i for batch in a for i in batch) == list(range(50))


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_roberta")
    tok = AutoTokenizer.from_pretrained("roberta-base")
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=130,
                        pad_token_id=tok.pad_token_id, bos_token_id=0, eos_token_id=2)
    RobertaForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def tiny_cfg(model_dir, epochs=3):
    return load_config(LORA_CFG, "rte", [f"model={model_dir}", f"epochs={epochs}",
                                         "batch_size=16", "max_seq_len=64", "lr=1e-3"])


def test_resume_matches_uninterrupted_run(tiny_model_dir, tmp_path, monkeypatch):
    cfg = tiny_cfg(tiny_model_dir)
    kw = {"data_limit": 64, "log": lambda *_: None}
    full = run_training(cfg, 0, tmp_path / "a", tmp_path / "a_ck", adapters_dir=tmp_path / "a_ad",
                        **kw)

    real_save = trainer.save_state

    def save_then_interrupt(*args, **kwargs):
        real_save(*args, **kwargs)
        if args[4] == 1:  # epoch counter
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "save_state", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_training(cfg, 0, tmp_path / "b", tmp_path / "b_ck", adapters_dir=tmp_path / "b_ad",
                     **kw)
    assert (tmp_path / "b_ck" / "lora_rte_r8_qv_s0.pt").exists()
    monkeypatch.setattr(trainer, "save_state", real_save)
    resumed = run_training(cfg, 0, tmp_path / "b", tmp_path / "b_ck",
                           adapters_dir=tmp_path / "b_ad", **kw)

    assert not (tmp_path / "b_ck" / "lora_rte_r8_qv_s0.pt").exists()  # removed on completion
    ha = json.loads((tmp_path / "a" / "runs" / "lora_rte_r8_qv_s0.json").read_text())["history"]
    hb = json.loads((tmp_path / "b" / "runs" / "lora_rte_r8_qv_s0.json").read_text())["history"]
    assert len(ha) == len(hb) == 3
    for x, y in zip(ha, hb, strict=True):
        assert x["train_loss"] == pytest.approx(y["train_loss"], rel=1e-3)
    assert full["best_epoch"] == resumed["best_epoch"]


def test_best_epoch_adapter_is_saved(tiny_model_dir, tmp_path):
    cfg = tiny_cfg(tiny_model_dir, epochs=2)
    row = run_training(cfg, 2, tmp_path / "r", tmp_path / "ck", data_limit=64,
                       adapters_dir=tmp_path / "adapters", log=lambda *_: None)
    path = tmp_path / "adapters" / "rte" / "seed2" / "lora_rte_r8_qv_s2.pt"
    blob = torch.load(path, weights_only=False)
    assert blob["best_epoch"] == row["best_epoch"]
    assert blob["best_metric"] == pytest.approx(row["best_metric"])
    names = blob["state_dict"].keys()
    assert names and all("lora_" in n or n.startswith("classifier.") for n in names)
    assert sum("lora_A" in n for n in names) == 4  # 2 layers x (q, v)


def test_ft_run_saves_no_adapter(tiny_model_dir, tmp_path):
    cfg = load_config(FT_CFG, "rte", [f"model={tiny_model_dir}", "epochs=1", "max_seq_len=64"])
    run_training(cfg, 0, tmp_path / "r", tmp_path / "ck", data_limit=32,
                 adapters_dir=tmp_path / "adapters", log=lambda *_: None)
    assert not (tmp_path / "adapters").exists()
