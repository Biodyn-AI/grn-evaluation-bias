"""Phase B (eval-side): score every method against every gold standard under each candidate set.

For each (dataset, method, gold_standard, candidate_set):
- Restrict predicted edges to candidate set
- Build positive labels from gold standard restricted to candidate set
- Compute AUPR, AUROC, top-K precision (K=50,100,500,1000), base rate, |U|

Outputs:
  outputs/eval/<tag>_metrics.parquet   one row per (method, gold_standard, candidate_set)
  outputs/eval/<tag>_summary.csv       pivot/summary

Usage:
  python scripts/eval_universe_aware.py --tag hESC
  python scripts/eval_universe_aware.py --tag hHep
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
PREP_ROOT = IMPL_ROOT / "outputs" / "prep"
BASELINES_ROOT = IMPL_ROOT / "outputs" / "baselines"
SCGPT_ROOT = IMPL_ROOT / "outputs" / "scgpt"
GENEFORMER_ROOT = IMPL_ROOT / "outputs" / "geneformer"
EVAL_ROOT = IMPL_ROOT / "outputs" / "eval"
NETWORKS_ROOT = IMPL_ROOT / "data" / "beeline" / "Networks"

# Gold standards per dataset
GOLD_STANDARDS = {
    "hESC": {
        "cell_type_chipseq": NETWORKS_ROOT / "human" / "hESC-ChIP-seq-network.csv",
        "nonspecific_chipseq": NETWORKS_ROOT / "human" / "Non-specific-ChIP-seq-network.csv",
        "string": NETWORKS_ROOT / "human" / "STRING-network.csv",
    },
    "hHep": {
        "cell_type_chipseq": NETWORKS_ROOT / "human" / "HepG2-ChIP-seq-network.csv",
        "nonspecific_chipseq": NETWORKS_ROOT / "human" / "Non-specific-ChIP-seq-network.csv",
        "string": NETWORKS_ROOT / "human" / "STRING-network.csv",
    },
}


def normalize_symbol(sym: str) -> str:
    return "".join(c for c in str(sym).upper() if c.isalnum() or c in "-.").strip()


def load_universe(tag: str):
    with open(PREP_ROOT / tag / f"{tag}_universe.json") as f:
        uni = json.load(f)
    # Reload mapped gene list from h5ad
    import anndata as ad
    adata = ad.read_h5ad(PREP_ROOT / tag / f"{tag}_raw.h5ad")
    genes = list(adata.var.index)
    # Recover TFs
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    from prep_beeline import load_tf_list
    T = sorted(load_tf_list() & set(genes))
    return uni, genes, T


def load_gold(path: Path, genes: set[str]) -> pd.DataFrame:
    """Return DataFrame columns (source, target) restricted to genes in our universe."""
    df = pd.read_csv(path)
    df.columns = ["source", "target"] + list(df.columns[2:])
    df["source"] = df["source"].map(normalize_symbol)
    df["target"] = df["target"].map(normalize_symbol)
    df = df[(df["source"].isin(genes)) & (df["target"].isin(genes))]
    df = df[df["source"] != df["target"]]
    return df.drop_duplicates(["source", "target"])


def build_candidate_mask(method_df: pd.DataFrame, candidate_set: str, T_set: set[str],
                          gold_tfs: set[str], gold_targets: set[str], genes: set[str]) -> np.ndarray:
    """Return boolean mask over method_df rows belonging to the candidate set."""
    src = method_df["source"]
    tgt = method_df["target"]
    if candidate_set == "all_pairs":
        # every (source, target) where both are in G and source != target
        return np.ones(len(method_df), dtype=bool)
    elif candidate_set == "tf_sources":
        return src.isin(T_set).values
    elif candidate_set == "tf_sources_targets":
        return (src.isin(gold_tfs) & tgt.isin(gold_targets)).values
    else:
        raise ValueError(candidate_set)


def evaluate_one(method_df: pd.DataFrame, gold_df: pd.DataFrame, candidate_set: str,
                  T_set: set[str], genes: set[str]) -> dict:
    """Compute metrics for one (method, gold, candidate)."""
    gold_tfs = set(gold_df["source"])
    gold_targets = set(gold_df["target"])
    # Restrict to candidate set
    mask = build_candidate_mask(method_df, candidate_set, T_set, gold_tfs, gold_targets, genes)
    sub = method_df.loc[mask, ["source", "target", "score"]]
    # Labels
    gold_pairs = set(zip(gold_df["source"], gold_df["target"]))
    y_true = np.fromiter(((s, t) in gold_pairs for s, t in zip(sub["source"], sub["target"])),
                          dtype=bool, count=len(sub))
    n_pos = int(y_true.sum())
    n = len(sub)
    base_rate = n_pos / max(n, 1)
    if n_pos == 0 or n_pos == n:
        return {
            "n_candidates": n, "n_positives": n_pos, "base_rate": base_rate,
            "aupr": np.nan, "auroc": np.nan,
            "precision_at_50": np.nan, "precision_at_100": np.nan,
            "precision_at_500": np.nan, "precision_at_1000": np.nan,
        }
    scores = sub["score"].values
    # Some methods (Pearson/Spearman) emit large continuous values; arboreto emits sparse importance.
    # Treat NaN scores as 0
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max)
    aupr = float(average_precision_score(y_true, scores))
    auroc = float(roc_auc_score(y_true, scores))
    # Top-K precision
    order = np.argsort(-scores)
    precision_at = {}
    for K in (50, 100, 500, 1000):
        if K <= n:
            top = y_true[order[:K]]
            precision_at[K] = float(top.sum() / K)
        else:
            precision_at[K] = np.nan
    return {
        "n_candidates": n, "n_positives": n_pos, "base_rate": base_rate,
        "aupr": aupr, "auroc": auroc,
        "precision_at_50": precision_at[50],
        "precision_at_100": precision_at[100],
        "precision_at_500": precision_at[500],
        "precision_at_1000": precision_at[1000],
    }


def method_files(tag: str) -> dict[str, Path]:
    """Return method_name -> parquet path for all available method outputs."""
    files = {}
    base = BASELINES_ROOT / tag
    if base.exists():
        for p in base.glob("*.parquet"):
            files[p.stem] = p
    sc = SCGPT_ROOT / tag
    if sc.exists():
        for p in sc.glob("*.parquet"):
            files[f"scgpt_{p.stem}"] = p
    gf = GENEFORMER_ROOT / tag
    if gf.exists():
        for p in gf.glob("*.parquet"):
            files[f"geneformer_{p.stem}"] = p
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    args = ap.parse_args()

    uni, genes, T = load_universe(args.tag)
    genes_set, T_set = set(genes), set(T)
    print(f"[{args.tag}] |G|={len(genes_set)}, |T|={len(T_set)}, |U|={uni['U_size']}", flush=True)

    methods = method_files(args.tag)
    print(f"[{args.tag}] methods available: {sorted(methods.keys())}", flush=True)
    if not methods:
        raise SystemExit(f"No method outputs for {args.tag}")

    rows = []
    for gs_name, gs_path in GOLD_STANDARDS[args.tag].items():
        gold = load_gold(gs_path, genes_set)
        print(f"[{args.tag}/{gs_name}] gold edges in universe: {len(gold)}", flush=True)
        for method_name, method_path in methods.items():
            df = pd.read_parquet(method_path)
            df["source"] = df["source"].map(normalize_symbol)
            df["target"] = df["target"].map(normalize_symbol)
            # Restrict to universe (both in G, source != target)
            df = df[df["source"].isin(genes_set) & df["target"].isin(genes_set)]
            df = df[df["source"] != df["target"]]
            for candidate_set in ("all_pairs", "tf_sources", "tf_sources_targets"):
                t0 = time.time()
                metrics = evaluate_one(df, gold, candidate_set, T_set, genes_set)
                metrics.update({"tag": args.tag, "method": method_name,
                                 "gold_standard": gs_name, "candidate_set": candidate_set,
                                 "elapsed_s": time.time() - t0})
                rows.append(metrics)
                print(f"  {method_name} | {gs_name} | {candidate_set}: "
                      f"AUPR={metrics['aupr']:.4g} AUROC={metrics['auroc']:.4g} "
                      f"P@50={metrics['precision_at_50']:.3f} n_pos={metrics['n_positives']} "
                      f"n_cand={metrics['n_candidates']} base_rate={metrics['base_rate']:.3g} "
                      f"({metrics['elapsed_s']:.1f}s)", flush=True)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    out = EVAL_ROOT / f"{args.tag}_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[{args.tag}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
