# Experiment log

## Current state / Next steps

- Stage: scaffold and paper extraction done; implementation not started.
- Next: install the CUDA PyTorch build, pin the environment, then data and metrics, LoRA layer,
  injection, training script, tests, smoke run, timing probe, benchmarks, core runs, rank
  ablation, projection ablation, report.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-11-15 | Repo root is the PyCharm project folder; single git repo created here. | Project location rule. |
| 2026-11-15 | Paper PDF is fetched by `scripts/fetch_paper.py` and ignored by git. | Avoid redistributing the PDF; the transcribed numbers are tracked in `paper/paper_results.yaml`. |
| 2026-11-15 | Table 2 has no dagger row for RoBERTa-base (only RoBERTa-large rows carry the dagger). The comparison column for the dagger row is therefore empty. | Paper content. |
| 2026-11-15 | Weight decay 0.1, AdamW betas (0.9, 0.98), eps 1e-6, grad clip 1.0, LoRA dropout 0, trained classification head. | Not specified in the paper; see `docs/deviations.md`. |
| 2026-11-15 | FT baseline uses lr 2e-5, batch size 16, 10 epochs for all tasks. | One fixed point of the Liu et al. grid, chosen before any run. |
| 2026-11-15 | The platform is Windows, so `setsid nohup` is replaced by a detached Python subprocess (`DETACHED_PROCESS`) started by the queue runner. | `setsid` is unavailable. |

## Open issues

None yet.
