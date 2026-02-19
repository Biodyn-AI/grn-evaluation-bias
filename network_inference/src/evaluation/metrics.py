from __future__ import annotations

import numpy as np


def aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    total_pos = int(labels.sum())
    if total_pos == 0:
        return 0.0
    labels = labels.astype(np.int64, copy=False)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precisions = tp / (tp + fp)
    recalls = tp / total_pos

    try:
        area = np.trapezoid(precisions, recalls)
    except AttributeError:
        area = np.trapz(precisions, recalls)
    return float(area)
