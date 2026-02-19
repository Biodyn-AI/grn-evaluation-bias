from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import scanpy as sc

from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def load_attention_scores(scores_path: str | Path, counts_path: str | Path) -> np.ndarray:
    score_sum = np.load(scores_path, mmap_mode="r")
    score_count = np.load(counts_path, mmap_mode="r")
    if score_sum.ndim != 2:
        raise ValueError(
            "Attention scores must be 2D. Run extraction with reduce_layers=true and reduce_heads=true."
        )
    ensure_mechinterp_path()
    from src.interpret.attention import finalize_attention_scores

    return finalize_attention_scores(score_sum, score_count)


def load_processed_anndata(path: str | Path) -> sc.AnnData:
    return sc.read_h5ad(Path(path))


def load_gene_names(path: str | Path) -> Tuple[sc.AnnData, np.ndarray]:
    adata = load_processed_anndata(path)
    return adata, adata.var_names.values
