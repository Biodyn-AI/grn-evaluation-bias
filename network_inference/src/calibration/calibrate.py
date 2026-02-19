from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from network_inference.src.calibration.isotonic import IsotonicModel, fit_isotonic
from network_inference.src.calibration.logistic import LogisticModel, fit_logistic
from network_inference.src.calibration.metrics import brier_score, expected_calibration_error, reliability_curve
from network_inference.src.data.loaders import load_attention_scores, load_processed_anndata
from network_inference.src.evaluation.ground_truth import true_edge_matrix
from network_inference.src.inference.candidates import candidate_config_from_sections, build_candidate_masks


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


def _sample_edges(
    scores: np.ndarray,
    labels: np.ndarray,
    max_edges: int | None,
    seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_edges is None or scores.size <= max_edges:
        return scores, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(scores.size, size=int(max_edges), replace=False)
    return scores[idx], labels[idx]


def _serialize_model(model: LogisticModel | IsotonicModel) -> Dict[str, object]:
    if isinstance(model, LogisticModel):
        return {"type": "logistic", "coef": model.coef, "intercept": model.intercept}
    return {"type": "isotonic", "thresholds": model.thresholds.tolist(), "values": model.values.tolist()}


def _apply_calibration_to_edges(
    edges_path: Path,
    model: LogisticModel | IsotonicModel,
    output_path: Path,
) -> int:
    edges = pd.read_csv(edges_path, sep="\t")
    scores = edges["score"].astype(float).values
    edges["confidence"] = model.predict(scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_path, sep="\t", index=False)
    return len(edges)


def run_calibration(config: Dict[str, object]) -> Dict[str, object]:
    paths = config["paths"]
    cal_cfg = dict(config.get("calibration", {}))
    network_cfg = dict(config.get("network", {}))
    omnipath_cfg = dict(config.get("omnipath", {}))

    scores = load_attention_scores(paths["attention_scores"], paths["attention_counts"])
    adata = load_processed_anndata(paths["processed_h5ad"])

    candidate_cfg = candidate_config_from_sections(network_cfg, cal_cfg, omnipath_cfg)
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

    true_mask = true_edge_matrix(
        gene_names_norm,
        paths,
        config.get("evaluation", {}).get("dorothea_confidence"),
        alias_map,
    )

    scores_flat = scores[candidate_mask]
    labels_flat = true_mask[candidate_mask].astype(np.int8)

    scores_fit, labels_fit = _sample_edges(
        scores_flat,
        labels_flat,
        cal_cfg.get("max_edges"),
        cal_cfg.get("sample_seed"),
    )

    method = cal_cfg.get("method", "isotonic")
    if method == "logistic":
        model = fit_logistic(
            scores_fit,
            labels_fit,
            max_iter=int(cal_cfg.get("max_iter", 500)),
            lr=float(cal_cfg.get("lr", 0.1)),
            l2=float(cal_cfg.get("l2", 0.0)),
        )
    elif method == "isotonic":
        model = fit_isotonic(scores_fit, labels_fit)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    probs = model.predict(scores_fit)
    curve = reliability_curve(probs, labels_fit, n_bins=int(cal_cfg.get("n_bins", 20)))

    report = {
        "method": method,
        "candidate_config": asdict(candidate_cfg),
        "n_edges_total": int(scores_flat.size),
        "n_edges_fit": int(scores_fit.size),
        "metrics": {
            "brier_score": brier_score(probs, labels_fit),
            "ece": expected_calibration_error(curve),
        },
        "reliability_curve": curve,
    }

    output_path = _resolve_optional_path(cal_cfg.get("output_path"), config)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    model_path = _resolve_optional_path(cal_cfg.get("model_path"), config)
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(_serialize_model(model), indent=2), encoding="utf-8")

    edges_path = _resolve_optional_path(cal_cfg.get("calibrated_edges_path"), config)
    if edges_path:
        _apply_calibration_to_edges(Path(paths["network_edges"]), model, edges_path)

    return report


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
