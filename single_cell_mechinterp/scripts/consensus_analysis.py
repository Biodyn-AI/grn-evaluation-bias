from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr, precision_recall_f1
from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.ravel()
    labels = labels.ravel().astype(bool)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    ranks = _rankdata(scores)
    sum_pos = ranks[labels].sum()
    return float((sum_pos - pos * (pos + 1) / 2) / (pos * neg))


def _precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> Tuple[float, float]:
    scores_flat = scores.ravel()
    labels_flat = labels.ravel().astype(bool)
    if scores_flat.size == 0:
        return 0.0, 0.0
    k = min(k, scores_flat.size)
    if k <= 0:
        return 0.0, 0.0
    top_idx = np.argpartition(-scores_flat, k - 1)[:k]
    tp = labels_flat[top_idx].sum()
    total_pos = labels_flat.sum()
    precision = float(tp / k) if k else 0.0
    recall = float(tp / total_pos) if total_pos else 0.0
    return precision, recall


def _load_gene_list(path: str | None, alias_map: Dict[str, str]) -> List[str]:
    if not path:
        return []
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    genes = []
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gene = line.split()[0]
        gene = canonical_symbol(gene, alias_map)
        if gene:
            genes.append(gene)
    return genes


def _resolve_candidates(
    gene_names_norm: np.ndarray,
    candidate_names: List[str],
) -> Tuple[List[str], List[int]]:
    gene_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(gene_names_norm):
        if name and name not in gene_to_idx:
            gene_to_idx[name] = idx

    if candidate_names:
        seen = set()
        ordered = []
        for name in candidate_names:
            if name in gene_to_idx and name not in seen:
                ordered.append(name)
                seen.add(name)
        names = ordered
    else:
        names = list(gene_to_idx.keys())

    indices = [gene_to_idx[name] for name in names]
    return names, indices


def _build_label_matrix(
    true_edges: pd.DataFrame,
    source_names: List[str],
    target_names: List[str],
) -> np.ndarray:
    source_index = {name: i for i, name in enumerate(source_names)}
    target_index = {name: i for i, name in enumerate(target_names)}
    labels = np.zeros((len(source_names), len(target_names)), dtype=np.int8)
    for _, row in true_edges.iterrows():
        i = source_index.get(row["source"])
        j = target_index.get(row["target"])
        if i is not None and j is not None:
            labels[i, j] = 1
    return labels


def _load_scores(matrices_dir: Path, probe: str) -> np.ndarray:
    scores_path = matrices_dir / f"{probe}.npy"
    if not scores_path.exists():
        raise FileNotFoundError(f"Score matrix not found: {scores_path}")
    return np.load(scores_path)


def _consensus_mean(scores_map: Dict[str, np.ndarray], probes: List[str]) -> np.ndarray:
    stack = np.stack([scores_map[probe] for probe in probes], axis=0)
    return stack.mean(axis=0).astype(np.float32)


def _consensus_rank_sum(
    scores_map: Dict[str, np.ndarray], probes: List[str], weights: Dict[str, float]
) -> np.ndarray:
    total = None
    for probe in probes:
        scores = scores_map[probe]
        flat = scores.ravel()
        ranks = _rankdata(-flat)
        if ranks.size > 1:
            ranks = 1.0 - (ranks - 1) / (ranks.size - 1)
        ranks = ranks.reshape(scores.shape).astype(np.float32)
        weight = float(weights.get(probe, 1.0))
        if total is None:
            total = weight * ranks
        else:
            total += weight * ranks
    if total is None:
        raise ValueError("No probes supplied for rank-sum consensus")
    return total.astype(np.float32)


def _topk_edges(
    scores: np.ndarray,
    source_indices: List[int],
    target_indices: List[int],
    top_k: int,
    remove_self: bool,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if top_k <= 0 or not source_indices or not target_indices:
        return edges
    target_indices_arr = np.array(target_indices, dtype=int)
    for source_idx in source_indices:
        row_scores = scores[source_idx, target_indices_arr]
        if remove_self:
            same_mask = target_indices_arr == source_idx
            if np.any(same_mask):
                row_scores = row_scores.copy()
                row_scores[same_mask] = -np.inf
        k = min(top_k, row_scores.size)
        if k <= 0:
            continue
        top_local = np.argpartition(-row_scores, k - 1)[:k]
        for target_pos in top_local:
            target_idx = target_indices_arr[target_pos]
            if not np.isfinite(row_scores[target_pos]):
                continue
            edges.add((source_idx, target_idx))
    return edges


def _consensus_intersection(
    scores_map: Dict[str, np.ndarray],
    probes: List[str],
    source_indices: List[int],
    target_indices: List[int],
    top_k: int,
    remove_self: bool,
    min_support: int,
) -> np.ndarray:
    support: Dict[tuple[int, int], int] = {}
    for probe in probes:
        edges = _topk_edges(
            scores_map[probe], source_indices, target_indices, top_k, remove_self
        )
        for edge in edges:
            support[edge] = support.get(edge, 0) + 1
    intersection_scores = np.zeros_like(next(iter(scores_map.values())), dtype=np.float32)
    for (src, tgt), count in support.items():
        if count >= min_support:
            intersection_scores[src, tgt] = 1.0
    return intersection_scores


def run_consensus(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    probe_cfg = cfg.get("probe_benchmark", {})
    probes = probe_cfg.get("probes", [])
    if not probes:
        raise ValueError("No probes configured for consensus analysis")

    output_dir = Path(paths.get("probe_output_dir", "outputs/probe_benchmark"))
    matrices_dir = Path(paths.get("probe_matrices_dir", output_dir / "matrices"))
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(paths["processed_h5ad"])
    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_set = set(gene_names_norm)

    candidate_sources = _load_gene_list(probe_cfg.get("candidate_sources_path"), alias_map)
    candidate_targets = _load_gene_list(probe_cfg.get("candidate_targets_path"), alias_map)
    source_names, source_indices = _resolve_candidates(gene_names_norm, candidate_sources)
    target_names, target_indices = _resolve_candidates(gene_names_norm, candidate_targets)

    source_mask = None
    target_mask = None
    if candidate_sources:
        source_mask = np.zeros(adata.n_vars, dtype=bool)
        source_mask[source_indices] = True
    if candidate_targets:
        target_mask = np.zeros(adata.n_vars, dtype=bool)
        target_mask[target_indices] = True

    references_cfg = probe_cfg.get("references")
    if not references_cfg:
        references_cfg = [
            {
                "name": "reference",
                "path": paths["dorothea_tsv"],
                "confidence_levels": cfg.get("evaluation", {}).get("dorothea_confidence"),
            }
        ]

    references = []
    for ref_cfg in references_cfg:
        true_edges = load_dorothea(
            ref_cfg["path"], confidence_levels=ref_cfg.get("confidence_levels")
        )
        true_edges = normalize_edges(true_edges, alias_map)
        true_edges = true_edges[
            true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
        ].drop_duplicates()
        labels = _build_label_matrix(true_edges, source_names, target_names)
        references.append((ref_cfg.get("name", "reference"), true_edges, labels))

    network_cfg = NetworkConfig(**cfg.get("network", {}))
    consensus_cfg = probe_cfg.get("consensus", {})
    weights = consensus_cfg.get("weights", {})
    min_support = int(consensus_cfg.get("support", len(probes)))

    scores_map: Dict[str, np.ndarray] = {}
    for probe in probes:
        scores_map[probe] = _load_scores(matrices_dir, probe)

    variants: Dict[str, np.ndarray] = {
        "consensus_mean": _consensus_mean(scores_map, probes),
        "consensus_rank_sum": _consensus_rank_sum(scores_map, probes, weights),
        "consensus_intersection_topk": _consensus_intersection(
            scores_map,
            probes,
            source_indices,
            target_indices,
            int(network_cfg.top_k or 0),
            network_cfg.remove_self,
            min_support,
        ),
    }

    metrics_rows = []
    for name, scores in variants.items():
        score_subset = scores[np.ix_(source_indices, target_indices)]
        edges = infer_edges(
            scores,
            adata.var_names,
            network_cfg,
            source_mask=source_mask,
            target_mask=target_mask,
        )
        edges = normalize_edges(edges, alias_map)
        edges = edges[
            edges["source"].isin(gene_set) & edges["target"].isin(gene_set)
        ].drop_duplicates()
        for ref_name, true_edges, labels in references:
            pr_metrics = precision_recall_f1(edges, true_edges)
            pr_at_k, rec_at_k = _precision_recall_at_k(
                score_subset, labels, int(probe_cfg.get("evaluation_top_k", 1000))
            )
            metrics_rows.append(
                {
                    "variant": name,
                    "reference": ref_name,
                    "precision": pr_metrics["precision"],
                    "recall": pr_metrics["recall"],
                    "f1": pr_metrics["f1"],
                    "aupr": aupr(score_subset.ravel(), labels.ravel()),
                    "auroc": _auroc(score_subset, labels),
                    "precision_at_k": pr_at_k,
                    "recall_at_k": rec_at_k,
                    "n_pred_edges": len(edges),
                    "n_true_edges": len(true_edges),
                    "n_candidates": score_subset.size,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "consensus_analysis.csv"
    metrics_df.to_csv(metrics_path, index=False)

    ablation_rows = []
    for base_name in ("consensus_mean", "consensus_rank_sum"):
        full_scores = variants[base_name]
        full_metrics = metrics_df[metrics_df["variant"] == base_name].set_index("reference")
        for probe in probes:
            subset = [p for p in probes if p != probe]
            if not subset:
                continue
            if base_name == "consensus_mean":
                scores = _consensus_mean(scores_map, subset)
            else:
                scores = _consensus_rank_sum(scores_map, subset, weights)
            score_subset = scores[np.ix_(source_indices, target_indices)]
            edges = infer_edges(
                scores,
                adata.var_names,
                network_cfg,
                source_mask=source_mask,
                target_mask=target_mask,
            )
            edges = normalize_edges(edges, alias_map)
            edges = edges[
                edges["source"].isin(gene_set) & edges["target"].isin(gene_set)
            ].drop_duplicates()
            for ref_name, true_edges, labels in references:
                pr_metrics = precision_recall_f1(edges, true_edges)
                pr_at_k, rec_at_k = _precision_recall_at_k(
                    score_subset, labels, int(probe_cfg.get("evaluation_top_k", 1000))
                )
                ablation_rows.append(
                    {
                        "variant": base_name,
                        "ablation_probe": probe,
                        "reference": ref_name,
                        "precision": pr_metrics["precision"],
                        "recall": pr_metrics["recall"],
                        "f1": pr_metrics["f1"],
                        "aupr": aupr(score_subset.ravel(), labels.ravel()),
                        "auroc": _auroc(score_subset, labels),
                        "precision_at_k": pr_at_k,
                        "recall_at_k": rec_at_k,
                    }
                )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = output_dir / "consensus_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)

    if not ablation_df.empty:
        delta_rows = []
        for base_name in ("consensus_mean", "consensus_rank_sum"):
            base_metrics = metrics_df[metrics_df["variant"] == base_name].set_index("reference")
            subset = ablation_df[ablation_df["variant"] == base_name]
            for _, row in subset.iterrows():
                ref = row["reference"]
                if ref not in base_metrics.index:
                    continue
                delta_rows.append(
                    {
                        "variant": base_name,
                        "ablation_probe": row["ablation_probe"],
                        "reference": ref,
                        "delta_aupr": row["aupr"] - base_metrics.loc[ref, "aupr"],
                        "delta_f1": row["f1"] - base_metrics.loc[ref, "f1"],
                    }
                )
        delta_df = pd.DataFrame(delta_rows)
        delta_df.to_csv(output_dir / "consensus_ablation_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze consensus probes and ablations")
    parser.add_argument("--config", default="configs/probe_benchmark.yaml")
    args = parser.parse_args()
    run_consensus(args.config)


if __name__ == "__main__":
    main()
