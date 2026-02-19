from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from network_inference.src.data.loaders import load_attention_scores, load_processed_anndata
from network_inference.src.inference.candidates import candidate_config_from_sections, build_candidate_masks
from network_inference.src.evaluation.ground_truth import true_edge_matrix
from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def _resolve_candidate_config(
    network_cfg: Dict[str, object],
    sweep_cfg: Dict[str, object],
    omnipath_cfg: Dict[str, object],
):
    return candidate_config_from_sections(network_cfg, sweep_cfg, omnipath_cfg)


def _candidate_matrix_mask(
    n_genes: int,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
    remove_self: bool,
) -> np.ndarray:
    mask = np.ones((n_genes, n_genes), dtype=bool)
    if source_mask is not None:
        mask &= source_mask[:, None]
    if target_mask is not None:
        mask &= target_mask[None, :]
    if remove_self:
        mask &= ~np.eye(n_genes, dtype=bool)
    return mask


def _true_edge_matrix(
    gene_names_norm: np.ndarray,
    paths: Dict[str, Path],
    confidence_levels: Iterable[str] | None,
    alias_map: Dict[str, str],
) -> np.ndarray:
    return true_edge_matrix(gene_names_norm, paths, confidence_levels, alias_map)


def _precision_recall_from_masks(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> Dict[str, float]:
    tp = int(np.logical_and(pred_mask, true_mask).sum())
    pred_total = int(pred_mask.sum())
    true_total = int(np.logical_and(true_mask, candidate_mask).sum())
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / true_total if true_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "predicted_edges": pred_total,
        "true_edges": true_total,
    }


def run_sweep(config: Dict[str, object]) -> Dict[str, object]:
    paths = config["paths"]
    sweep_cfg = dict(config.get("sweep", {}))
    network_cfg = dict(config.get("network", {}))
    omnipath_cfg = dict(config.get("omnipath", {}))

    scores = load_attention_scores(paths["attention_scores"], paths["attention_counts"])
    adata = load_processed_anndata(paths["processed_h5ad"])

    candidate_cfg = _resolve_candidate_config(network_cfg, sweep_cfg, omnipath_cfg)
    source_mask, target_mask, alias_map, gene_names_norm = build_candidate_masks(
        adata,
        paths,
        candidate_cfg,
        config.get("evaluation", {}).get("dorothea_confidence"),
        intercell_cfg=config.get("intercell", {}),
        expression_cfg=config.get("expression_filter", {}),
    )

    remove_self = bool(network_cfg.get("remove_self", True))
    candidate_mask = _candidate_matrix_mask(scores.shape[0], source_mask, target_mask, remove_self)
    if not candidate_mask.any():
        raise ValueError("Candidate mask is empty; adjust candidate filters.")

    true_mask = _true_edge_matrix(
        gene_names_norm,
        paths,
        config.get("evaluation", {}).get("dorothea_confidence"),
        alias_map,
    )

    result: Dict[str, object] = {
        "candidate_config": asdict(candidate_cfg),
        "candidate_edges": int(candidate_mask.sum()),
    }

    scores_flat = scores[candidate_mask]
    labels_flat = true_mask[candidate_mask].astype(np.int8)

    ensure_mechinterp_path()
    from src.eval.metrics import aupr

    result["aupr"] = float(aupr(scores_flat, labels_flat))

    percentile_values = sweep_cfg.get("percentiles", [])
    percentile_results: List[Dict[str, float]] = []
    for percentile in percentile_values:
        threshold = float(np.percentile(scores_flat, percentile))
        pred_mask = candidate_mask & (scores >= threshold)
        metrics = _precision_recall_from_masks(pred_mask, true_mask, candidate_mask)
        metrics["percentile"] = float(percentile)
        metrics["threshold"] = threshold
        percentile_results.append(metrics)
    result["percentile_sweep"] = percentile_results

    top_k_values = sweep_cfg.get("top_k_values", [])
    top_k_results: List[Dict[str, float]] = []
    if top_k_values:
        ensure_mechinterp_path()
        from src.network.infer import NetworkConfig, infer_edges

        for k in top_k_values:
            network_cfg_local = NetworkConfig(
                threshold_percentile=network_cfg.get("threshold_percentile", 95.0),
                top_k=int(k),
                remove_self=remove_self,
            )
            edges = infer_edges(scores, adata.var_names, network_cfg_local, source_mask, target_mask)
            pred_mask = np.zeros_like(candidate_mask, dtype=bool)
            if not edges.empty:
                name_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}
                for _, row in edges.iterrows():
                    src = name_to_idx.get(row["source"])
                    tgt = name_to_idx.get(row["target"])
                    if src is None or tgt is None:
                        continue
                    pred_mask[src, tgt] = True
            metrics = _precision_recall_from_masks(pred_mask, true_mask, candidate_mask)
            metrics["top_k"] = int(k)
            top_k_results.append(metrics)
    result["top_k_sweep"] = top_k_results

    output_path = _resolve_optional_path(sweep_cfg.get("output_path"), config)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_dumps(result), encoding="utf-8")

    return result


def json_dumps(payload: Dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2)


def _resolve_optional_path(path_value: str | Path | None, config: Dict[str, object]) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    base_dir = config.get("_config_dir")
    if base_dir:
        return (Path(base_dir) / path).resolve()
    return path
