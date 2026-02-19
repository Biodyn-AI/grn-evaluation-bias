from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_gene_names
from src.eval.metrics import aupr
from src.network.infer import NetworkConfig
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


def _build_true_mask(
    true_edges: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    n_genes: int,
) -> np.ndarray:
    mask = np.zeros((n_genes, n_genes), dtype=bool)
    for _, row in true_edges.iterrows():
        i = gene_to_idx.get(row["source"])
        j = gene_to_idx.get(row["target"])
        if i is None or j is None:
            continue
        mask[i, j] = True
    return mask


def _load_scores(matrices_dir: Path, probe: str) -> np.ndarray:
    scores_path = matrices_dir / f"{probe}.npy"
    if not scores_path.exists():
        raise FileNotFoundError(f"Score matrix not found: {scores_path}")
    return np.load(scores_path)


def run_sweep(
    config_path: str,
    output_path: str | None,
    probe_override: List[str] | None,
) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    probe_cfg = cfg.get("probe_benchmark", {})

    probes = probe_cfg.get("probes", [])
    if probe_override:
        probes = probe_override
    if not probes:
        raise ValueError("No probes configured for threshold sweep")

    output_dir = Path(paths.get("probe_output_dir", "outputs/probe_benchmark"))
    matrices_dir = Path(paths.get("probe_matrices_dir", output_dir / "matrices"))
    sweep_path = Path(output_path) if output_path else output_dir / "probe_threshold_sweep.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(paths["processed_h5ad"])
    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_set = set(gene_names_norm)
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names_norm) if name}

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

    reference_data = []
    for ref_cfg in references_cfg:
        true_edges = load_dorothea(
            ref_cfg["path"], confidence_levels=ref_cfg.get("confidence_levels")
        )
        true_edges = true_edges.copy()
        true_edges["source"] = (
            true_edges["source"].astype(str).str.strip().str.upper().map(lambda x: alias_map.get(x, x))
        )
        true_edges["target"] = (
            true_edges["target"].astype(str).str.strip().str.upper().map(lambda x: alias_map.get(x, x))
        )
        true_edges = true_edges[
            true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
        ].drop_duplicates()
        labels = _build_label_matrix(true_edges, source_names, target_names)
        true_mask = _build_true_mask(true_edges, gene_to_idx, adata.n_vars)
        reference_data.append(
            (ref_cfg.get("name", "reference"), true_edges, true_mask, int(true_mask.sum()), labels)
        )

    sweep_cfg = probe_cfg.get("threshold_sweep", {})
    top_k_values = [int(v) for v in sweep_cfg.get("top_k_values", [])]
    percentile_values = [float(v) for v in sweep_cfg.get("percentile_values", [])]

    network_cfg = NetworkConfig(**cfg.get("network", {}))
    mask_modes = sweep_cfg.get("mask_modes", ["masked"])

    rows = []
    for probe in probes:
        scores = _load_scores(matrices_dir, probe)
        if scores.shape[0] != adata.n_vars or scores.shape[1] != adata.n_vars:
            raise ValueError(f"Score matrix shape mismatch for {probe}: {scores.shape}")

        score_subset = scores[np.ix_(source_indices, target_indices)]
        for ref_name, true_edges, true_mask, total_pos, labels in reference_data:
            base_aupr = aupr(score_subset.ravel(), labels.ravel())
            base_auroc = _auroc(score_subset, labels)

            for mask_mode in mask_modes:
                if mask_mode == "masked":
                    active_sources = source_indices or list(range(adata.n_vars))
                    active_targets = target_indices or list(range(adata.n_vars))
                else:
                    active_sources = list(range(adata.n_vars))
                    active_targets = list(range(adata.n_vars))

                scores_sub = scores[np.ix_(active_sources, active_targets)]
                true_sub = true_mask[np.ix_(active_sources, active_targets)]
                mask_matrix = np.ones(scores_sub.shape, dtype=bool)
                if network_cfg.remove_self:
                    target_pos = {idx: pos for pos, idx in enumerate(active_targets)}
                    for row_pos, gene_idx in enumerate(active_sources):
                        col_pos = target_pos.get(gene_idx)
                        if col_pos is not None:
                            mask_matrix[row_pos, col_pos] = False

                for top_k in top_k_values:
                    if scores_sub.size == 0:
                        pred_count = 0
                        tp = 0
                    else:
                        scores_for_topk = scores_sub.copy()
                        scores_for_topk[~mask_matrix] = -np.inf
                        k = min(top_k, scores_for_topk.shape[1])
                        if k <= 0:
                            pred_count = 0
                            tp = 0
                        else:
                            top_idx = np.argpartition(-scores_for_topk, k - 1, axis=1)[:, :k]
                            row_idx = np.arange(scores_for_topk.shape[0])[:, None]
                            valid = np.isfinite(scores_for_topk[row_idx, top_idx])
                            pred_count = int(valid.sum())
                            if pred_count:
                                tp = int(true_sub[row_idx, top_idx][valid].sum())
                            else:
                                tp = 0
                    precision = float(tp / pred_count) if pred_count else 0.0
                    recall = float(tp / total_pos) if total_pos else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    rows.append(
                        {
                            "probe": probe,
                            "reference": ref_name,
                            "mask_mode": mask_mode,
                            "threshold_type": "top_k",
                            "threshold_value": top_k,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "aupr": base_aupr,
                            "auroc": base_auroc,
                            "n_pred_edges": pred_count,
                            "n_true_edges": len(true_edges),
                            "n_candidates": score_subset.size,
                            "n_sources": len(source_indices),
                            "n_targets": len(target_indices),
                            "active_sources": len(active_sources),
                            "active_targets": len(active_targets),
                            "candidate_masked": bool(candidate_sources or candidate_targets),
                        }
                    )

                for percentile in percentile_values:
                    if scores_sub.size == 0:
                        pred_count = 0
                        tp = 0
                    else:
                        masked_scores = scores_sub[mask_matrix]
                        if masked_scores.size == 0:
                            pred_count = 0
                            tp = 0
                        else:
                            threshold = float(np.percentile(masked_scores, percentile))
                            pred_mask = (scores_sub >= threshold) & mask_matrix
                            pred_count = int(pred_mask.sum())
                            tp = int((true_sub & pred_mask).sum()) if pred_count else 0
                    precision = float(tp / pred_count) if pred_count else 0.0
                    recall = float(tp / total_pos) if total_pos else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    rows.append(
                        {
                            "probe": probe,
                            "reference": ref_name,
                            "mask_mode": mask_mode,
                            "threshold_type": "percentile",
                            "threshold_value": percentile,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "aupr": base_aupr,
                            "auroc": base_auroc,
                            "n_pred_edges": pred_count,
                            "n_true_edges": len(true_edges),
                            "n_candidates": score_subset.size,
                            "n_sources": len(source_indices),
                            "n_targets": len(target_indices),
                            "active_sources": len(active_sources),
                            "active_targets": len(active_targets),
                            "candidate_masked": bool(candidate_sources or candidate_targets),
                        }
                    )

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(sweep_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep network thresholds for probe matrices")
    parser.add_argument("--config", default="configs/probe_benchmark.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--probes", nargs="*", default=None)
    args = parser.parse_args()
    run_sweep(args.config, args.output, args.probes)


if __name__ == "__main__":
    main()
