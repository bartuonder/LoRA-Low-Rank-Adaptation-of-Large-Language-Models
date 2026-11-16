| Task | Metric | Paper (published) | Paper (dagger row) | Mine: mean ± std | Mine: median | Δ (median − paper) | Seeds | Likely reason (hypothesis) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| SST-2 | Accuracy | 95.1 ± 0.2 | n/a | 94.8 ± 0.2 | 94.8 | -0.3 | 5 | Hypothesis: unspecified settings (weight decay, optimizer betas, adapter initialisation scale) or bf16 autocast differ from the original runs. |
| MRPC | Accuracy | 89.7 ± 0.7 | n/a | 89.2 ± 0.5 | 89.2 | -0.5 | 5 | Agreement despite the pre-trained start. |
| CoLA | Matthews corr. | 63.4 ± 1.2 | n/a | 63.0 ± 1.0 | 63.1 | -0.3 | 5 | No MNLI start is involved; close agreement is expected. |
| RTE | Accuracy | 86.6 ± 0.7 | n/a | 78.6 ± 0.5 | 78.7 | -7.9 | 5 | Hypothesis: the published run starts from an MNLI-adapted checkpoint, which is known to help the small RTE task considerably; here the start is the pre-trained model. |
| STS-B | Pearson corr. | 91.5 ± 0.2 | n/a | 91.1 ± 0.1 | 91.0 | -0.5 | 5 | Hypothesis: the published run starts from an MNLI-adapted checkpoint; here the start is the pre-trained model. Regression head initialisation may also matter. |
| Mean of the five tasks | – | 85.3 | n/a | – | 83.4 | -1.9 | – | Dominated by the tasks that start from an MNLI checkpoint in the paper. |
