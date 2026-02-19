from __future__ import annotations

from typing import Tuple

import numpy as np


def expression_masks(
    adata,
    min_mean_expr: float | None = None,
    min_frac_cells: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min_mean_expr is None and min_frac_cells is None:
        mask = np.ones(adata.n_vars, dtype=bool)
        return mask, np.zeros(adata.n_vars, dtype=float), np.zeros(adata.n_vars, dtype=float)

    matrix = adata.X
    if hasattr(matrix, "mean"):
        mean_expr = np.asarray(matrix.mean(axis=0)).ravel()
    else:
        mean_expr = np.asarray(matrix).mean(axis=0)

    if min_frac_cells is not None:
        if hasattr(matrix, "getnnz"):
            frac_expr = np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
        else:
            frac_expr = np.asarray((matrix > 0).mean(axis=0)).ravel()
    else:
        frac_expr = np.zeros(adata.n_vars, dtype=float)

    mask = np.ones(adata.n_vars, dtype=bool)
    if min_mean_expr is not None:
        mask &= mean_expr >= float(min_mean_expr)
    if min_frac_cells is not None:
        mask &= frac_expr >= float(min_frac_cells)

    return mask, mean_expr, frac_expr
