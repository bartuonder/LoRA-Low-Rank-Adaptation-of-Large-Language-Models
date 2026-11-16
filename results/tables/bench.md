Measured on NVIDIA GeForce RTX 4070 SUPER (torch 2.11.0+cu128), commit `ea81122`.

**Trainable parameters** (RoBERTa-base with a two-class head; the head is randomly initialised and always trained, and is not part of the paper's count)

| Method | LoRA A/B | Head | Trainable total | Share of full model |
| :--- | ---: | ---: | ---: | ---: |
| Full fine-tuning | – | 124,647,170 | 124,647,170 | 100.0 % |
| LoRA r=1 | 36,864 | 592,130 | 628,994 | 0.50 % |
| LoRA r=2 | 73,728 | 592,130 | 665,858 | 0.53 % |
| LoRA r=4 | 147,456 | 592,130 | 739,586 | 0.59 % |
| LoRA r=8 | 294,912 | 592,130 | 887,042 | 0.71 % |

**Adapter versus full checkpoint (file size)**

| Artifact | Size | Relative |
| :--- | ---: | ---: |
| Full checkpoint (state dict, fp32) | 475.6 MiB | 1× |
| LoRA r=1, A and B only | 0.16 MiB | 3056× smaller |
| LoRA r=1, A, B and head | 2.42 MiB | 197× smaller |
| LoRA r=2, A and B only | 0.30 MiB | 1605× smaller |
| LoRA r=2, A, B and head | 2.56 MiB | 186× smaller |
| LoRA r=4, A and B only | 0.58 MiB | 824× smaller |
| LoRA r=4, A, B and head | 2.84 MiB | 168× smaller |
| LoRA r=8, A and B only | 1.14 MiB | 417× smaller |
| LoRA r=8, A, B and head | 3.40 MiB | 140× smaller |

**Inference latency, merged versus unmerged** (median over many iterations after warm-up, synchronised timing; batch size 1 is launch-bound and noisy)

| Batch | Seq. len. | Precision | Unmerged (ms) | Merged (ms) | Speed-up |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | bf16 | 12.88 | 8.16 | 1.58× |
| 1 | 128 | fp32 | 7.99 | 6.09 | 1.31× |
| 16 | 128 | bf16 | 10.44 | 8.86 | 1.18× |
| 16 | 128 | fp32 | 22.79 | 21.94 | 1.04× |
| 32 | 128 | bf16 | 17.10 | 15.82 | 1.08× |
| 32 | 128 | fp32 | 45.44 | 43.94 | 1.03× |

Maximum absolute logit difference, merged vs unmerged (fp32): 7.45e-08.

**Peak GPU memory** (bf16 autocast, fp32 parameters, AdamW for training; LoRA inference uses merged weights)

| Mode | Batch | Seq. len. | LoRA (MiB) | Full FT (MiB) | LoRA saving |
| :--- | ---: | ---: | ---: | ---: | ---: |
| train | 16 | 128 | 1271 | 2476 | 49 % |
| inference | 16 | 128 | 541 | 730 | 26 % |
| train | 32 | 128 | 1853 | 3253 | 43 % |
| inference | 32 | 128 | 587 | 771 | 24 % |
