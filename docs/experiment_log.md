# Experiment log

## Current state / Next steps

- Stage: implementation, tests (27 passing) and smoke run done; timing probe running.
- Next: `scripts/estimate_compute.py` -> `docs/compute_plan.md`, apply the budget ladder to
  `configs/matrix.yaml` if needed, benchmarks (`make bench`, GPU idle), then core runs
  (`make core-detached`), rank ablation, projection ablation, report and README.
- Resume: `python scripts/launch_detached.py --matrix core` (skips runs already in
  `results/runs.csv`).

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-11-15 | Repo root is the PyCharm project folder; single git repo created here. | Project location rule. |
| 2026-11-15 | Paper PDF is fetched by `scripts/fetch_paper.py` and ignored by git. | Avoid redistributing the PDF; the transcribed numbers are tracked in `paper/paper_results.yaml`. |
| 2026-11-15 | Table 2 has no dagger row for RoBERTa-base (only RoBERTa-large rows carry the dagger). The comparison column for the dagger row is therefore empty. | Paper content. |
| 2026-11-15 | Weight decay 0.1, AdamW betas (0.9, 0.98), eps 1e-6, grad clip 1.0, LoRA dropout 0, trained classification head. | Not specified in the paper; see `docs/deviations.md`. |
| 2026-11-15 | FT baseline uses lr 2e-5, batch size 16, 10 epochs for all tasks. | One fixed point of the Liu et al. grid, chosen before any run. |
| 2026-11-16 | LoRA A initialised from a Gaussian with std 1/sqrt(3 x in_features). | The paper says "random Gaussian" without a variance; this matches the variance of the reference kaiming-uniform initialisation and does not depend on r. |
| 2026-11-16 | Every finished LoRA run exports its best-epoch A/B and head weights to `adapters/<task>/seed<k>/<run_id>.pt` (gitignored). | Requested addition. Rank and projection ablations share (task, seed) with the core runs, so the file name carries the run id. The best snapshot is stored in the resume checkpoint, so an interrupted run keeps it. |
| 2026-11-16 | YAML-style overrides parse `1e-4` as a float (PyYAML reads it as a string). | Found by a test; would have broken `--set lr=1e-4`. |
| 2026-11-15 | The platform is Windows, so `setsid nohup` is replaced by a detached Python subprocess (`DETACHED_PROCESS`) started by the queue runner. | `setsid` is unavailable. |

## Open issues

None yet.
