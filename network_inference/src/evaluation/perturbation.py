from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from network_inference.src.evaluation.custom_edges import evaluate_edge_list, normalize_edge_list


def evaluate_perturbations(
    pred_edges_path: str | Path,
    perturbation_tsv: str | Path,
    processed_h5ad: str | Path,
    hgnc_alias_tsv: str | Path | None,
    source_col: str = "perturbed_gene",
    target_col: str = "affected_gene",
    perturbation_col: str = "perturbation",
    mapping_cfg: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if not perturbation_tsv:
        raise ValueError("perturbation.edges_tsv is required for perturbation evaluation")
    df = pd.read_csv(perturbation_tsv, sep="\t")
    if perturbation_col not in df.columns:
        raise ValueError(f"Perturbation file must include {perturbation_col} column")
    perturbation_col_internal = perturbation_col
    if perturbation_col == source_col or perturbation_col == target_col:
        perturbation_col_internal = "perturbation"
        required = df[[source_col, target_col]].dropna()
        perturb_edges = required.copy()
        perturb_edges[perturbation_col_internal] = df.loc[required.index, perturbation_col]
    else:
        required = df[[source_col, target_col, perturbation_col]].dropna()
        perturb_edges = required[[source_col, target_col, perturbation_col]].copy()
    perturb_edges.columns = ["source", "target", perturbation_col_internal]

    perturb_edges_norm, _ = normalize_edge_list(
        perturb_edges,
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg=mapping_cfg,
    )
    pred_edges = pd.read_csv(pred_edges_path, sep="\t")[["source", "target"]].drop_duplicates()
    pred_edges_norm, _ = normalize_edge_list(
        pred_edges,
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg=mapping_cfg,
    )

    pred_set = {(row["source"], row["target"]) for _, row in pred_edges_norm.iterrows()}
    per_perturbation = []
    for perturbation, group in perturb_edges_norm.groupby(perturbation_col_internal):
        truth_set = {(row["source"], row["target"]) for _, row in group.iterrows()}
        if not truth_set:
            continue
        hits = len(truth_set & pred_set)
        recall = hits / len(truth_set) if truth_set else 0.0
        per_perturbation.append(
            {
                "perturbation": perturbation,
                "truth_edges": len(truth_set),
                "hits": hits,
                "recall": recall,
            }
        )

    overall = evaluate_edge_list(
        pred_edges_path,
        perturbation_tsv,
        processed_h5ad,
        hgnc_alias_tsv,
        truth_source_col=source_col,
        truth_target_col=target_col,
        mapping_cfg=mapping_cfg,
    )
    avg_recall = float(sum(item["recall"] for item in per_perturbation) / len(per_perturbation)) if per_perturbation else 0.0

    return {
        "overall_metrics": overall,
        "per_perturbation_recall": per_perturbation,
        "avg_recall": avg_recall,
    }
