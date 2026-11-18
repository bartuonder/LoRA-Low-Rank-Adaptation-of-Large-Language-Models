# Experiment log

## Current state / Next steps

- All stages finished (2026-11-18 19:30): core (50 runs), rank ablation (100 runs including the
  shared r=8 runs) and projection ablation (60 runs including the shared Wq+Wv runs); 170 of 170
  runs, no failures, no diverged runs. Results, tables and figures are committed.
- Budget: measured training time 17.8 h (core), 43.4 h (rank ablation, new ranks) and 4.8 h
  (projection ablation, new variants); no reduction step of the ladder was applied.
- Checks done: 27 tests pass, lint clean, `scripts/make_report.py` regenerates all tables,
  figures and README blocks without changes, no remotes, all commits authored by the repository
  owner, no forbidden content in tracked files or commit messages.
- Nothing is pending. To redo a stage: `python scripts/launch_detached.py --matrix all` skips
  finished runs (`results/runs.csv`); an interrupted run resumes from `checkpoints/`.

## Notes

- Compute: SST-2 runs took longer than the probe-based plan (LoRA 2.4 h, FT 0.5 h per run);
  small tasks ran faster. `scripts/queue_status.py` overstates the remainder because it counts
  keys shared between groups more than once.
- Monitoring sessions were interrupted three times by a host low-memory guard on a 15.6 GB
  machine (largest processes: the training child at about 1.6 GB and the IDE at about 2 GB). The
  queue runner and training children were never affected.
- Earlier in the sweep, unit tests were run on the GPU once while the queue was active; it did
  not affect any result. Tests run with `CUDA_VISIBLE_DEVICES=` during sweeps.

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

## Notes

- 2026-11-16 00:27: the unit tests in `tests/test_training.py` were run once while the queue was
  active, so a tiny model briefly shared the GPU with the first training run. This only cost GPU
  time; results are unaffected. Tests are run with `CUDA_VISIBLE_DEVICES=` (CPU) during sweeps.

## Open issues

None yet.
