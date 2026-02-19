from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import scanpy as sc

from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def evaluate_dorothea(
    edges_path: str | Path,
    processed_h5ad: str | Path,
    dorothea_tsv: str | Path,
    hgnc_alias_tsv: str | Path | None,
    confidence_levels: Iterable[str] | None,
) -> Dict[str, float]:
    ensure_mechinterp_path()
    from src.eval.dorothea import load_dorothea
    from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges, normalize_gene_names
    from src.eval.metrics import precision_recall_f1

    pred_edges_df = pd.read_csv(edges_path, sep="\t")
    pred_edges_df = pred_edges_df[["source", "target"]].drop_duplicates()

    true_edges = load_dorothea(dorothea_tsv, confidence_levels=confidence_levels)

    adata = sc.read_h5ad(Path(processed_h5ad))
    alias_map = load_hgnc_alias_map(hgnc_alias_tsv)
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_set = set(gene_names_norm)

    pred_edges_df = normalize_edges(pred_edges_df, alias_map)
    pred_edges_df = pred_edges_df[
        pred_edges_df["source"].isin(gene_set) & pred_edges_df["target"].isin(gene_set)
    ].drop_duplicates()
    true_edges = normalize_edges(true_edges, alias_map)
    true_edges = true_edges[
        true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
    ].drop_duplicates()

    return precision_recall_f1(pred_edges_df, true_edges)
