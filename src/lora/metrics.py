import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef

# Scores are reported on a 0-100 scale, as in the paper.


def compute_metric(name: str, preds: np.ndarray, labels: np.ndarray) -> float:
    if name == "accuracy":
        return 100.0 * float((preds == labels).mean())
    if name == "matthews_correlation":
        return 100.0 * float(matthews_corrcoef(labels, preds))
    if name == "pearson":
        if np.std(preds) == 0:  # constant predictions: correlation undefined, count as 0
            return 0.0
        return 100.0 * float(pearsonr(preds, labels)[0])
    raise ValueError(f"unknown metric {name!r}")
