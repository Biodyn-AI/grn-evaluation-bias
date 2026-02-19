from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IsotonicModel:
    thresholds: np.ndarray
    values: np.ndarray

    def predict(self, scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.thresholds, scores, side="right")
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicModel:
    order = np.argsort(scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order].astype(float)

    block_values = []
    block_counts = []
    block_max_scores = []

    for score, label in zip(scores_sorted, labels_sorted):
        block_values.append(label)
        block_counts.append(1)
        block_max_scores.append(score)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total = block_values[-2] * block_counts[-2] + block_values[-1] * block_counts[-1]
            count = block_counts[-2] + block_counts[-1]
            block_values[-2] = total / count
            block_counts[-2] = count
            block_max_scores[-2] = block_max_scores[-1]
            block_values.pop()
            block_counts.pop()
            block_max_scores.pop()

    return IsotonicModel(
        thresholds=np.array(block_max_scores, dtype=float),
        values=np.array(block_values, dtype=float),
    )
