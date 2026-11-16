- Model: `roberta-base`; LoRA on query and value projections, r = 8, alpha = 8, LoRA dropout 0.0, classification head trained.
- Optimiser: AdamW, linear schedule with 0.06 warm-up ratio, weight decay 0.1, betas (0.9, 0.98), epsilon 1e-06, gradient clipping 1; maximum sequence length 512; bf16 autocast.
- Seeds: 0, 1, 2, 3, 4.
- Hardware and libraries: NVIDIA GeForce RTX 4070 SUPER; torch 2.11.0+cu128, transformers 5.23.0.
- Per-task values below: LoRA from Table 9 of the paper, FT from the fixed point of the RoBERTa fine-tuning grid ([`configs/lora_roberta_base.yaml`](configs/lora_roberta_base.yaml), [`configs/ft_roberta_base.yaml`](configs/ft_roberta_base.yaml)).

| Task | LoRA lr | LoRA batch | LoRA epochs | FT lr | FT batch | FT epochs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| SST-2 | 0.0005 | 16 | 60 | 2e-05 | 16 | 10 |
| MRPC | 0.0004 | 16 | 30 | 2e-05 | 16 | 10 |
| CoLA | 0.0004 | 32 | 80 | 2e-05 | 16 | 10 |
| RTE | 0.0005 | 32 | 80 | 2e-05 | 16 | 10 |
| STS-B | 0.0004 | 16 | 40 | 2e-05 | 16 | 10 |
