"""Phase B3: baselines (Pearson, Spearman, GENIE3, GRNBoost2, random) over the full universe U = T x (G\\{self}).

For each method, writes a Parquet table with columns (source, target, score) covering all pairs in U.
Universe matches the prep step exactly so eval-time comparisons are apples-to-apples.

Usage:
  python scripts/run_baselines.py --tag hESC --method pearson
  python scripts/run_baselines.py --tag hESC --method spearman
  python scripts/run_baselines.py --tag hESC --method random
  python scripts/run_baselines.py --tag hESC --method grnboost2 --n-estimators 100
  python scripts/run_baselines.py --tag hESC --method genie3 --n-estimators 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
PREP_ROOT = IMPL_ROOT / "outputs" / "prep"
OUT_ROOT = IMPL_ROOT / "outputs" / "baselines"


def load_data(tag: str):
    import anndata as ad
    adata = ad.read_h5ad(PREP_ROOT / tag / f"{tag}_raw.h5ad")
    with open(PREP_ROOT / tag / f"{tag}_universe.json") as f:
        uni = json.load(f)
    # cells x genes; ensure float for stats
    if sp.issparse(adata.X):
        X = adata.X.toarray().astype(np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)
    genes = list(adata.var.index)
    # TFs in this dataset
    from prep_beeline import load_tf_list, normalize_symbol  # type: ignore
    T = sorted(load_tf_list() & set(genes))
    assert len(T) == uni["T_size"], f"TF count mismatch: {len(T)} vs {uni['T_size']}"
    return X, genes, T


def write_scores(out_path: Path, src: np.ndarray, tgt: np.ndarray, score: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"source": src, "target": tgt, "score": score})
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} rows)", flush=True)


def run_correlation(tag: str, method: str, seed: int = 42):
    X, genes, T = load_data(tag)
    n_cells, n_genes = X.shape
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tf_idx = np.array([gene_to_idx[t] for t in T])

    # For Pearson: standardize columns, then score[i,j] = (X_i^T X_j) / n_cells
    Xc = X - X.mean(axis=0, keepdims=True)
    # rank for spearman
    if method == "spearman":
        from scipy.stats import rankdata
        X_for_corr = np.apply_along_axis(rankdata, 0, X).astype(np.float32)
        X_for_corr = X_for_corr - X_for_corr.mean(axis=0, keepdims=True)
    else:
        X_for_corr = Xc
    # Standardize
    norms = np.linalg.norm(X_for_corr, axis=0)
    norms[norms == 0] = 1.0
    Xn = X_for_corr / norms

    # Score matrix: TF (rows) x targets (cols)
    print(f"  computing {method} matrix [{len(T)} x {n_genes}] ...", flush=True)
    t0 = time.time()
    scores = (Xn[:, tf_idx].T @ Xn).astype(np.float32)  # |T| x |G|
    # Absolute value (signed correlation maps to |corr| as regulatory strength proxy, BEELINE convention)
    scores = np.abs(scores)
    # Zero out self-pairs
    for i, ti in enumerate(tf_idx):
        scores[i, ti] = 0.0
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    # Flatten to (source, target, score)
    src = np.repeat(np.asarray(T), n_genes)
    tgt = np.tile(np.asarray(genes), len(T))
    flat_scores = scores.flatten()
    # Filter out self-pairs (score=0 from above) — but we keep them with score=-inf? Simpler: drop where src == tgt
    keep = src != tgt
    write_scores(OUT_ROOT / tag / f"{method}.parquet", src[keep], tgt[keep], flat_scores[keep])


def run_random(tag: str, seed: int = 42):
    X, genes, T = load_data(tag)
    rng = np.random.default_rng(seed)
    n_genes = len(genes)
    src = np.repeat(np.asarray(T), n_genes)
    tgt = np.tile(np.asarray(genes), len(T))
    score = rng.random(len(src), dtype=np.float32)
    keep = src != tgt
    write_scores(OUT_ROOT / tag / "random.parquet", src[keep], tgt[keep], score[keep])


def _fit_one_target(target_idx: int, X: np.ndarray, tf_idx: np.ndarray, genes: list[str],
                    T: list[str], method: str, n_estimators: int, seed: int):
    """Fit one regressor for target gene, return (target_gene, list_of_(tf, importance))."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    target_gene = genes[target_idx]
    y = X[:, target_idx]
    # Exclude self if target is a TF
    use_tf_idx = tf_idx[tf_idx != target_idx]
    if len(use_tf_idx) == 0:
        return target_gene, []
    Xf = X[:, use_tf_idx]
    if method == "genie3":
        reg = RandomForestRegressor(
            n_estimators=n_estimators, max_features="sqrt",
            n_jobs=1, random_state=seed,
        )
    else:  # grnboost2
        reg = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=0.05,
            max_depth=3, subsample=0.8, max_features=0.3,
            random_state=seed,
        )
    reg.fit(Xf, y)
    imp = reg.feature_importances_
    use_tf_names = [T[np.where(tf_idx == i)[0][0]] for i in use_tf_idx]
    return target_gene, list(zip(use_tf_names, imp.astype(np.float32).tolist()))


def run_arboreto(tag: str, method: str, n_estimators: int = 100, seed: int = 42,
                  n_jobs: int = 4):
    """GENIE3 (RF) or GRNBoost2 (GBM) via sklearn + joblib."""
    from joblib import Parallel, delayed
    X, genes, T = load_data(tag)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tf_idx = np.array([gene_to_idx[t] for t in T])
    n_cells, n_genes = X.shape
    print(f"  sklearn-{method} with {n_estimators} estimators on {n_cells} cells x {n_genes} genes, |T|={len(T)}, n_jobs={n_jobs} ...", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10, batch_size=8)(
        delayed(_fit_one_target)(j, X, tf_idx, genes, T, method, n_estimators, seed + j)
        for j in range(n_genes)
    )
    print(f"    fit done in {time.time()-t0:.1f}s; assembling output ...", flush=True)
    src_list, tgt_list, sc_list = [], [], []
    for target_gene, pairs in results:
        for tf_name, imp in pairs:
            src_list.append(tf_name)
            tgt_list.append(target_gene)
            sc_list.append(imp)
    result = pd.DataFrame({"source": src_list, "target": tgt_list, "score": np.asarray(sc_list, dtype=np.float32)})
    result = result[result["source"] != result["target"]]
    out = OUT_ROOT / tag / f"{method}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"  wrote {out} ({len(result)} rows) in total {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    ap.add_argument("--method", required=True,
                    choices=["pearson", "spearman", "random", "genie3", "grnboost2"])
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    if args.method in ("pearson", "spearman"):
        run_correlation(args.tag, args.method, args.seed)
    elif args.method == "random":
        run_random(args.tag, args.seed)
    else:
        run_arboreto(args.tag, args.method, args.n_estimators, args.seed, args.n_jobs)
    print(f"[{args.tag}/{args.method}] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
