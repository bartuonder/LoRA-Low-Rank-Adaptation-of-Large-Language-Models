# Deviations from the paper

All differences between this project and Hu et al. (2021) are listed here. Hyperparameters
that the paper does specify (Table 9) are used unchanged and are never adjusted after
seeing results.

## Planned deviations

| Item | Paper value | This project | Reason |
|---|---|---|---|
| Starting point for MRPC, RTE, STS-B | LoRA modules initialised from the best MNLI checkpoint (Appendix D.1); full fine-tuning baseline also starts from an MNLI model | Pre-trained `roberta-base` for all tasks | MNLI is out of scope (compute). Largest expected source of difference, mainly on RTE and MRPC. Table 2 has no dagger row for RoBERTa-base, so there is no published number for this exact setting. |
| Full fine-tuning baseline | Row quoted from Liu et al. (2019): per-task grid over learning rate and batch size, MNLI start for small tasks, median of 5 seeds | One fixed point of the Liu et al. grid (lr 2e-5, batch size 16, 10 epochs, weight decay 0.1) for every task, pre-trained start, seeds 0-4 | No per-task tuning; shows what the same pipeline gives without a sweep. Reported as "my FT run" next to the published row. |
| Evaluation split | Not stated for the RoBERTa section; the FT row equals the dev-set numbers of Liu et al. | GLUE dev (validation) split | GLUE test labels are hidden. |
| Hardware and precision | NVIDIA Tesla V100 | RTX 4070 SUPER, bf16 autocast with fp32 parameters | Available hardware. Numerical differences are expected. |

## Settings not specified in the paper (choices made here)

| Item | Paper value | This project | Reason |
|---|---|---|---|
| Weight decay | not given for RoBERTa | 0.1 | GLUE value of Liu et al. (2019), Table 10, whose recipe the paper follows. |
| AdamW betas / epsilon | not given | (0.9, 0.98) / 1e-6 | RoBERTa pre-training values (Liu et al., Table 9); same for LoRA and FT. |
| Gradient clipping | not given | max norm 1.0 | Common default; same for LoRA and FT. |
| LoRA dropout | not given for RoBERTa | 0.0 | No dropout inside the adapter. Model dropout stays at the RoBERTa default of 0.1. |
| Classification head | not stated | trained (randomly initialised) | The head has no pre-trained weights. Its parameters (about 0.59M) are not part of the 0.3M LoRA parameter count in Table 2, which equals 12 layers x 2 matrices x 2 x 768 x 8 = 294,912. |
| Mixed precision / padding | not given | bf16 autocast, dynamic padding, max length 512 | Padding does not change the result; it only avoids wasted compute. |

## Rank ablation

| Item | Paper value | This project | Reason |
|---|---|---|---|
| alpha across ranks | alpha is set to the first r tried and not tuned (Section 4.1); Table 9 uses alpha = r = 8 | alpha = r for r in {1, 2, 4, 8} (alpha/r = 1) | Keeps the scaling ratio constant so that only the rank changes. |

## Compute-budget reductions

Filled in after the timing probe (see `docs/compute_plan.md`).

| Step | Applied | Effect |
|---|---|---|
| 1. Drop SST-2 from the rank ablation | pending | |
| 2. Rank-ablation seeds 5 -> 3 for CoLA | pending | |
| 3. Core seeds 5 -> 3 for SST-2 | pending | |
