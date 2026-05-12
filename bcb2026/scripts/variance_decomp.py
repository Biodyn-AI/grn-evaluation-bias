"""Phase B7: ANOVA-style variance decomposition of AUPR.

Reads outputs/eval/*_metrics.parquet and decomposes log10(AUPR) variance across
factors (candidate_set | gold_standard | method | tag). Outputs partial eta-squared.

Usage:
  python scripts/variance_decomp.py [--y log_aupr|aupr]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
def manual_anova_typeII(df: pd.DataFrame, y: str, factors: list[str]) -> pd.DataFrame:
    """Type-II SS via successive OLS without statsmodels (avoids scipy incompat)."""
    from numpy.linalg import lstsq
    Y = df[y].values.astype(np.float64)
    n = len(Y)
    # Build full model design matrix from all factors (one-hot, drop reference per factor)
    def design(factors_list):
        cols = [np.ones((n, 1))]
        for f in factors_list:
            cats = sorted(df[f].astype(str).unique())[1:]  # drop reference
            for c in cats:
                cols.append((df[f].astype(str) == c).values.reshape(-1, 1).astype(np.float64))
        return np.concatenate(cols, axis=1)
    Xfull = design(factors)
    ssr_full = ((Y - Xfull @ lstsq(Xfull, Y, rcond=None)[0]) ** 2).sum()
    sst = ((Y - Y.mean()) ** 2).sum()
    rows = []
    for f in factors:
        others = [g for g in factors if g != f]
        Xred = design(others) if others else np.ones((n, 1))
        ssr_red = ((Y - Xred @ lstsq(Xred, Y, rcond=None)[0]) ** 2).sum()
        ss = ssr_red - ssr_full
        rows.append({"factor": f, "sum_sq_typeII": ss, "pct_total_ss": ss / sst * 100})
    rows.append({"factor": "Residual_full_model", "sum_sq_typeII": ssr_full, "pct_total_ss": ssr_full / sst * 100})
    rows.append({"factor": "Total", "sum_sq_typeII": sst, "pct_total_ss": 100.0})
    return pd.DataFrame(rows)

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent / "outputs" / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", default="log_aupr", choices=["log_aupr", "aupr"])
    args = ap.parse_args()

    frames = []
    for f in EVAL_ROOT.glob("*_metrics.parquet"):
        df = pd.read_parquet(f)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics in {EVAL_ROOT}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["aupr"]).copy()
    df["log_aupr"] = np.log10(df["aupr"].clip(lower=1e-8))
    # Coarse method family
    def family(m):
        if m.startswith("scgpt"): return "scgpt"
        if m.startswith("geneformer"): return "geneformer"
        if m in {"genie3","grnboost2"}: return "tree"
        if m in {"pearson","spearman"}: return "corr"
        if m == "random": return "random"
        return "other"
    df["method_family"] = df["method"].map(family)

    aov = manual_anova_typeII(df, args.y, ["candidate_set", "gold_standard", "tag", "method_family"])
    print("ANOVA Type-II on", args.y)
    print(aov.round(4).to_string(index=False))
    out = EVAL_ROOT / f"variance_decomp_{args.y}.csv"
    aov.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
