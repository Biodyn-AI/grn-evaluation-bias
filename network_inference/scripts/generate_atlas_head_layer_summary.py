#!/usr/bin/env python3
"""Generate head/layer atlas correlation summary for evidence TSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize head/layer evidence vs atlas mean correlations "
            "for scores/counts arrays."
        )
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path("single_cell_mechinterp/outputs/atlas"),
        help="Root directory containing atlas datasets.",
    )
    parser.add_argument(
        "--evidence-top-2k",
        type=Path,
        default=Path("network_inference/outputs/head_layer_evidence_top_2k.tsv"),
        help="Head/layer evidence TSV for top-2k edges.",
    )
    parser.add_argument(
        "--evidence-top-10k",
        type=Path,
        default=Path("network_inference/outputs/head_layer_evidence_top_10k.tsv"),
        help="Head/layer evidence TSV for top-10k edges.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("network_inference/outputs/head_layer_atlas_correlations.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["immune", "kidney", "lung", "external_krasnow_lung"],
        help="Atlas dataset subdirectories to include.",
    )
    return parser.parse_args()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_corr(a, b)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return _safe_corr(ar, br)


def load_evidence_means(
    path: Path, layers: int, heads: int
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    sums = np.zeros((layers, heads), dtype=np.float64)
    counts = np.zeros((layers, heads), dtype=np.int64)

    for raw in df.get("top_head_layers", []):
        evidence = json.loads(raw)
        for entry in evidence:
            layer = int(entry["layer"])
            head = int(entry["head"])
            if 0 <= layer < layers and 0 <= head < heads:
                sums[layer, head] += float(entry["score"])
                counts[layer, head] += 1

    means = np.zeros_like(sums)
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask]
    return means, counts


def atlas_means(array: np.ndarray) -> Dict[str, np.ndarray]:
    layers, heads = array.shape[:2]
    mean_all = np.zeros((layers, heads), dtype=np.float64)
    mean_nonzero = np.zeros((layers, heads), dtype=np.float64)
    zero_fraction = np.zeros((layers, heads), dtype=np.float64)

    for layer in range(layers):
        for head in range(heads):
            data = array[layer, head]
            total = data.size
            if total == 0:
                continue
            total_sum = float(data.sum(dtype=np.float64))
            nonzero = int(np.count_nonzero(data))
            mean_all[layer, head] = total_sum / total
            if nonzero > 0:
                mean_nonzero[layer, head] = total_sum / nonzero
            zero_fraction[layer, head] = 1.0 - (nonzero / total)

    return {
        "mean_all": mean_all,
        "mean_nonzero": mean_nonzero,
        "zero_fraction": zero_fraction,
    }


def iter_atlas_arrays(atlas_root: Path, dataset: str) -> Iterable[Tuple[str, Path]]:
    base = atlas_root / dataset
    yield "scores", base / "attention_scores_head_layer.npy"
    yield "counts", base / "attention_counts_head_layer.npy"


def main() -> None:
    args = parse_args()
    records = []

    if not args.atlas_root.exists():
        raise FileNotFoundError(f"Atlas root not found: {args.atlas_root}")

    for dataset in args.datasets:
        for metric, path in iter_atlas_arrays(args.atlas_root, dataset):
            if not path.exists():
                continue
            array = np.load(path, mmap_mode="r")
            if array.ndim != 4:
                print(f"Skipping {path} (expected 4D; got {array.ndim}D)")
                continue
            layers, heads = array.shape[:2]

            evidence_sets = {
                "top_2k": args.evidence_top_2k,
                "top_10k": args.evidence_top_10k,
            }
            atlas_stats = atlas_means(array)

            for evidence_name, evidence_path in evidence_sets.items():
                if not evidence_path.exists():
                    continue
                evidence_means, evidence_counts = load_evidence_means(
                    evidence_path, layers, heads
                )
                evidence_flat = evidence_means.reshape(-1)
                evidence_nonzero_heads = int(np.count_nonzero(evidence_counts))
                zero_fraction_mean = float(atlas_stats["zero_fraction"].mean())

                for mean_type in ("mean_all", "mean_nonzero"):
                    atlas_flat = atlas_stats[mean_type].reshape(-1)
                    records.append(
                        {
                            "dataset": dataset,
                            "atlas_metric": metric,
                            "evidence_set": evidence_name,
                            "mean_type": mean_type,
                            "pearson": pearson(evidence_flat, atlas_flat),
                            "spearman": spearman(evidence_flat, atlas_flat),
                            "evidence_nonzero_heads": evidence_nonzero_heads,
                            "atlas_zero_fraction_mean": zero_fraction_mean,
                        }
                    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
