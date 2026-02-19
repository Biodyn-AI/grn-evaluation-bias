from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def true_edge_matrix(
    gene_names_norm: np.ndarray,
    paths: Dict[str, Path],
    confidence_levels: Iterable[str] | None,
    alias_map: Dict[str, str],
) -> np.ndarray:
    ensure_mechinterp_path()
    from src.eval.dorothea import load_dorothea
    from src.eval.gene_symbols import normalize_edges

    true_edges = load_dorothea(paths["dorothea_tsv"], confidence_levels=confidence_levels)
    true_edges = normalize_edges(true_edges, alias_map)

    gene_to_idx = {name: idx for idx, name in enumerate(gene_names_norm)}
    mat = np.zeros((len(gene_names_norm), len(gene_names_norm)), dtype=bool)
    for _, row in true_edges.iterrows():
        src = gene_to_idx.get(row["source"])
        tgt = gene_to_idx.get(row["target"])
        if src is None or tgt is None:
            continue
        mat[src, tgt] = True
    return mat
