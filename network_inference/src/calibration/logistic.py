from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticModel:
    coef: float
    intercept: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        logits = self.coef * scores + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic(
    scores: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 500,
    lr: float = 0.1,
    l2: float = 0.0,
) -> LogisticModel:
    scores = scores.astype(float)
    labels = labels.astype(float)
    coef = 0.0
    intercept = np.log(labels.mean() / max(1e-12, 1 - labels.mean())) if labels.mean() not in (0, 1) else 0.0

    for _ in range(max_iter):
        logits = coef * scores + intercept
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - labels
        grad_coef = (error * scores).mean() + l2 * coef
        grad_intercept = error.mean()
        coef -= lr * grad_coef
        intercept -= lr * grad_intercept

    return LogisticModel(coef=coef, intercept=intercept)
