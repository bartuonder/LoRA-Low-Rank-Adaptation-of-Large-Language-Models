"""Cross-check paper/paper_results.yaml against the text of the paper PDF."""
import re
import sys
from pathlib import Path

import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["MNLI", "SST-2", "MRPC", "CoLA", "QNLI", "QQP", "RTE", "STS-B"]
# line prefix in the extracted PDF text -> YAML row; the next token is the parameter column
ROWS = {
    "ft": "RoBbase (FT)*",
    "bitfit": "RoBbase (BitFit)*",
    "adapter_d_0_3m": "RoBbase (AdptD)* 0.3M",
    "adapter_d_0_9m": "RoBbase (AdptD)* 0.9M",
    "lora": "RoBbase (LoRA) 0.3M",
}


def parse_cell(tok: str) -> tuple[float, float | None]:
    m = re.fullmatch(r"(\d+\.\d)(?:±(\d*\.\d+|\d+))?", tok)
    if not m:
        raise ValueError(tok)
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


def main() -> int:
    pdf = ROOT / "paper" / "lora_2106.09685.pdf"
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)
    spec = yaml.safe_load((ROOT / "paper" / "paper_results.yaml").read_text(encoding="utf-8"))
    bad = 0
    for key, prefix in ROWS.items():
        line = next((ln for ln in text.splitlines() if ln.startswith(prefix)), None)
        if line is None:
            print(f"row not found: {prefix}")
            bad += 1
            continue
        toks = line[len(prefix):].split()
        if toks and toks[0].endswith("M"):
            toks = toks[1:]  # drop the trainable-parameter column when not in the prefix
        cells = [parse_cell(t) for t in toks[: len(TASKS) + 1]]
        row = spec["roberta_base"][key]
        for task, (val, std) in zip(TASKS, cells[:-1], strict=True):
            exp = row["scores"][task]
            if exp["value"] != val or exp.get("std") != std:
                print(f"MISMATCH {key} {task}: yaml={exp} pdf={(val, std)}")
                bad += 1
        if row["avg"] != cells[-1][0]:
            print(f"MISMATCH {key} avg: yaml={row['avg']} pdf={cells[-1][0]}")
            bad += 1

    hp = spec["hyperparameters_table9_base"]["per_task"]
    block = text[text.index("RoBERTa base\nLoRA"):]

    def vals(label: str) -> list[str]:
        ln = next(x for x in block.splitlines() if x.startswith(label))
        return ln[len(label):].split()[:8]

    bs = [int(v) for v in vals("Batch Size")]
    ep = [int(v) for v in vals("# Epochs")]
    lr = [float(v) for v in vals("Learning Rate")]
    for i, task in enumerate(TASKS):
        got = (bs[i], ep[i], lr[i])
        exp = (hp[task]["batch_size"], hp[task]["epochs"], hp[task]["lr"])
        if got != exp:
            print(f"MISMATCH Table 9 {task}: yaml={exp} pdf={got}")
            bad += 1
    print("paper numbers verified" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
