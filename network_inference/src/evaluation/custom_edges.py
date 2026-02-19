from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from network_inference.src.data.mapping import load_hgnc_alias_map_extended, map_edge_table
from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def load_edge_list(path: str | Path, source_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Edge file must include {source_col} and {target_col} columns")
    edges = df[[source_col, target_col]].dropna().copy()
    edges.columns = ["source", "target"]
    return edges.drop_duplicates()


def normalize_edge_list(
    edges: pd.DataFrame,
    processed_h5ad: str | Path,
    hgnc_alias_tsv: str | Path | None,
    mapping_cfg: Dict[str, object] | None = None,
    gene_names_norm: Iterable[str] | None = None,
    alias_map: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, set[str]]:
    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_edges, normalize_gene_names

    if gene_names_norm is None or alias_map is None:
        gene_names_norm, alias_map = _load_gene_names(
            processed_h5ad,
            hgnc_alias_tsv,
            mapping_cfg,
        )
    gene_set = set(gene_names_norm)

    edges = map_edge_table(edges, mapping_cfg, alias_map)
    edges = normalize_edges(edges, alias_map)
    edges = edges[edges["source"].isin(gene_set) & edges["target"].isin(gene_set)].drop_duplicates()
    return edges, gene_set


def evaluate_edge_list(
    pred_edges_path: str | Path,
    truth_edges_path: str | Path,
    processed_h5ad: str | Path,
    hgnc_alias_tsv: str | Path | None,
    truth_source_col: str = "source",
    truth_target_col: str = "target",
    mapping_cfg: Dict[str, object] | None = None,
    remove_self: bool = True,
) -> Dict[str, float]:
    ensure_mechinterp_path()
    from network_inference.src.evaluation.metrics import aupr
    from src.eval.metrics import precision_recall_f1

    gene_names_norm, alias_map = _load_gene_names(
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg,
    )
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_norm)}

    pred_edges = pd.read_csv(pred_edges_path, sep="\t")
    pred_cols = ["source", "target"] + (["score"] if "score" in pred_edges.columns else [])
    pred_edges = pred_edges[pred_cols].drop_duplicates()
    pred_edges, _ = normalize_edge_list(
        pred_edges,
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg=mapping_cfg,
        gene_names_norm=gene_names_norm,
        alias_map=alias_map,
    )
    if "score" in pred_edges.columns:
        pred_edges = (
            pred_edges.groupby(["source", "target"], as_index=False)["score"].max()
        )

    truth_edges = load_edge_list(truth_edges_path, truth_source_col, truth_target_col)
    truth_edges, _ = normalize_edge_list(
        truth_edges,
        processed_h5ad,
        hgnc_alias_tsv,
        mapping_cfg=mapping_cfg,
        gene_names_norm=gene_names_norm,
        alias_map=alias_map,
    )

    prf = precision_recall_f1(
        pred_edges[["source", "target"]],
        truth_edges[["source", "target"]],
    )

    scores = np.zeros((len(gene_names_norm), len(gene_names_norm)), dtype=np.float32)
    for _, row in pred_edges.iterrows():
        src = gene_to_idx.get(row["source"])
        tgt = gene_to_idx.get(row["target"])
        if src is None or tgt is None:
            continue
        score = float(row["score"]) if "score" in row else 1.0
        if score > scores[src, tgt]:
            scores[src, tgt] = score

    true_mask = np.zeros((len(gene_names_norm), len(gene_names_norm)), dtype=bool)
    for _, row in truth_edges.iterrows():
        src = gene_to_idx.get(row["source"])
        tgt = gene_to_idx.get(row["target"])
        if src is None or tgt is None:
            continue
        true_mask[src, tgt] = True

    candidate_mask = np.ones((len(gene_names_norm), len(gene_names_norm)), dtype=bool)
    if remove_self:
        candidate_mask &= ~np.eye(len(gene_names_norm), dtype=bool)
    scores_flat = scores[candidate_mask]
    labels_flat = true_mask[candidate_mask].astype(np.int8)
    aupr_value = float(aupr(scores_flat, labels_flat))

    return {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "aupr": aupr_value,
    }


def _load_gene_names(
    processed_h5ad: str | Path,
    hgnc_alias_tsv: str | Path | None,
    mapping_cfg: Dict[str, object] | None,
) -> Tuple[np.ndarray, Dict[str, str]]:
    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_gene_names

    adata = sc.read_h5ad(Path(processed_h5ad))
    extra_cols = None
    if mapping_cfg:
        extra_cols = mapping_cfg.get("hgnc_extra_alias_cols")
    alias_map = load_hgnc_alias_map_extended(hgnc_alias_tsv, extra_cols)
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    return gene_names_norm, alias_map
