from __future__ import annotations

from typing import List, Dict

import numpy as np


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 20,
) -> List[Dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    results: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            results.append(
                {"bin_lo": float(lo), "bin_hi": float(hi), "count": 0, "mean_prob": 0.0, "mean_label": 0.0}
            )
            continue
        mean_prob = float(probs[mask].mean())
        mean_label = float(labels[mask].mean())
        results.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": count,
                "mean_prob": mean_prob,
                "mean_label": mean_label,
            }
        )
    return results


def expected_calibration_error(curve: List[Dict[str, float]]) -> float:
    total = sum(bin_info["count"] for bin_info in curve)
    if total == 0:
        return 0.0
    ece = 0.0
    for bin_info in curve:
        weight = bin_info["count"] / total if total else 0.0
        ece += weight * abs(bin_info["mean_prob"] - bin_info["mean_label"])
    return float(ece)
