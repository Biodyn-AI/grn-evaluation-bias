from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc

from src.utils.config import load_config


def _to_dense(X: np.ndarray) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _apply_sampling(
    adata: sc.AnnData,
    max_cells: int | None,
    sample_mode: str | None,
    sample_seed: int | None,
    project_seed: int | None,
) -> sc.AnnData:
    if sample_mode and not max_cells:
        raise ValueError("sample_cells requires max_cells to be set")
    if not max_cells:
        return adata
    max_cells = int(max_cells)
    if sample_mode:
        if sample_mode not in {"random", "bootstrap"}:
            raise ValueError(f"Unsupported sample_cells: {sample_mode}")
        seed = sample_seed if sample_seed is not None else project_seed
        rng = np.random.default_rng(seed)
        if sample_mode == "random":
            size = min(max_cells, adata.n_obs)
            indices = rng.choice(adata.n_obs, size=size, replace=False)
        else:
            indices = rng.choice(adata.n_obs, size=max_cells, replace=True)
        return adata[indices].copy()
    if adata.n_obs > max_cells:
        return adata[:max_cells].copy()
    return adata


def _pearson_correlation(X: np.ndarray) -> np.ndarray:
    n_cells = X.shape[0]
    if n_cells < 2:
        return np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    X = X / std
    corr = (X.T @ X) / max(n_cells - 1, 1)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def run_baseline(config_path: str, output_path: str | None, use_abs: bool | None) -> None:
    cfg = load_config(config_path)
    project_seed = cfg.get("project", {}).get("seed")
    paths = cfg["paths"]
    probe_cfg = cfg.get("probe_benchmark", {})

    adata = sc.read_h5ad(paths["processed_h5ad"])
    adata = _apply_sampling(
        adata,
        probe_cfg.get("max_cells"),
        probe_cfg.get("sample_cells"),
        probe_cfg.get("sample_seed"),
        project_seed,
    )

    X = _to_dense(adata.X)
    corr = _pearson_correlation(X)
    use_abs = probe_cfg.get("baseline_use_abs", True) if use_abs is None else use_abs
    if use_abs:
        corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)

    output_dir = Path(paths.get("probe_output_dir", "outputs/probe_benchmark"))
    matrices_dir = Path(paths.get("probe_matrices_dir", output_dir / "matrices"))
    matrices_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path else matrices_dir / "coexpression.npy"
    np.save(output_path, corr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a correlation baseline matrix for probe evaluation")
    parser.add_argument("--config", default="configs/probe_benchmark.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--abs", dest="use_abs", action="store_true")
    parser.add_argument("--no-abs", dest="use_abs", action="store_false")
    parser.set_defaults(use_abs=None)
    args = parser.parse_args()
    run_baseline(args.config, args.output, args.use_abs)


if __name__ == "__main__":
    main()
