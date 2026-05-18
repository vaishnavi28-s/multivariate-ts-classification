"""
threshold.py
------------
Decision boundary threshold tuning on the validation set.

Rather than using the default 0.5 threshold, we search for the threshold
that maximises F1 on the validation set. This is critical for imbalanced
industrial data where the minority class (paper_problem, ~8.5% of events)
must be recalled reliably.

Threshold is tuned per fold and saved as part of the artefact — it is
applied at inference time without recomputation.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Searches thresholds from 0.05 to 0.95 in steps of 0.01.
    Returns the threshold that maximises F1 on the provided set.

    Args:
        y_true: ground truth binary labels
        y_prob: predicted probabilities for the positive class

    Returns:
        Best threshold (float between 0.05 and 0.95)
    """
    best_thr, best_f1 = 0.5, -1.0

    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return float(best_thr)
