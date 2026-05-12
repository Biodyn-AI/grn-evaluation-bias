"""Phase B5 (simplified): AUROC 95% CI using DeLong's method + paired AUPR bootstrap on subsample.

AUROC null = 0.5 (exact). If AUROC + 1.96*SE < 0.5, the method is significantly below random.

For AUPR, paired bootstrap on a 100k-row subsample (200 resamples) of (method, random) gives a
95% CI on Delta = AUPR_method - AUPR_random. If upper bound < 0, significantly below random.

Outputs:
  outputs/eval/<tag>_auroc_ci.parquet
  outputs/eval/<tag>_paired_bootstrap.parquet (subsample-based)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
EVAL_ROOT = IMPL_ROOT / "outputs" / "eval"
BASELINES_ROOT = IMPL_ROOT / "outputs" / "baselines"

sys.path.insert(0, str(HERE))
from eval_universe_aware import (
    GOLD_STANDARDS, load_gold, load_universe, method_files, normalize_symbol,
    build_candidate_mask,
)


def auroc_delong_se(y_true, y_score):
    """DeLong's variance estimate for AUROC."""
    y_true = np.asarray(y_true, dtype=bool)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    # Mid-rank for ties: use scipy rankdata
    from scipy.stats import rankdata
    order = np.concatenate([pos, neg])
    ranks = rankdata(order)
    rp = ranks[:n_pos]
    rn = ranks[n_pos:]
    # placement values for positives = (ranks - 1 - (in-pos rank)) / n_neg
    pos_within = rankdata(pos)
    neg_within = rankdata(neg)
    V10 = (rp - pos_within) / n_neg
    V01 = 1.0 - (rn - neg_within) / n_pos
    auroc = V10.mean()
    s10 = V10.var(ddof=1) / n_pos
    s01 = V01.var(ddof=1) / n_neg
    se = np.sqrt(s10 + s01)
    return float(auroc), float(se)


def paired_aupr_bootstrap(y_true, m_scores, r_scores, B=200, sub_n=100_000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=min(sub_n, n))
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        m = average_precision_score(yt, m_scores[idx])
        r = average_precision_score(yt, r_scores[idx])
        diffs.append(m - r)
    diffs = np.asarray(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--sub-n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidate-set", default="tf_sources",
                    help="Run only on this candidate set (default tf_sources)")
    args = ap.parse_args()

    uni, genes, T = load_universe(args.tag)
    genes_set, T_set = set(genes), set(T)
    methods = method_files(args.tag)
    print(f"[{args.tag}] |G|={len(genes_set)} |T|={len(T_set)}; methods={sorted(methods.keys())}", flush=True)

    random_path = BASELINES_ROOT / args.tag / "random.parquet"
    rdf = pd.read_parquet(random_path)
    rdf["source"] = rdf["source"].map(normalize_symbol)
    rdf["target"] = rdf["target"].map(normalize_symbol)

    auroc_rows = []
    boot_rows = []
    for gs_name, gs_path in GOLD_STANDARDS[args.tag].items():
        gold = load_gold(gs_path, genes_set)
        gold_pairs = set(zip(gold["source"], gold["target"]))
        gold_tfs, gold_targets = set(gold["source"]), set(gold["target"])
        rmask = build_candidate_mask(rdf, args.candidate_set, T_set, gold_tfs, gold_targets, genes_set)
        rsub = rdf.loc[rmask, ["source", "target", "score"]].reset_index(drop=True)
        y_true = np.fromiter(((s, t) in gold_pairs for s, t in zip(rsub["source"], rsub["target"])),
                              dtype=bool, count=len(rsub))
        r_scores = np.nan_to_num(rsub["score"].values.astype(np.float32))
        r_aupr = float(average_precision_score(y_true, r_scores))
        r_auroc, r_se = auroc_delong_se(y_true, r_scores)
        print(f"\n[{args.tag}/{gs_name}/{args.candidate_set}] random AUPR={r_aupr:.4g} AUROC={r_auroc:.4f}±{r_se:.4f} n_pos={int(y_true.sum())}/{len(rsub)}", flush=True)

        for method_name, method_path in methods.items():
            t0 = time.time()
            df = pd.read_parquet(method_path)
            df["source"] = df["source"].map(normalize_symbol)
            df["target"] = df["target"].map(normalize_symbol)
            df = df[df["source"].isin(genes_set) & df["target"].isin(genes_set)]
            df = df[df["source"] != df["target"]]
            mmask = build_candidate_mask(df, args.candidate_set, T_set, gold_tfs, gold_targets, genes_set)
            msub = df.loc[mmask, ["source", "target", "score"]]
            # Align to random rows
            msub = msub.set_index(["source", "target"]).reindex(
                list(zip(rsub["source"], rsub["target"]))
            ).reset_index()
            m_scores = np.nan_to_num(msub["score"].values.astype(np.float32))
            m_aupr = float(average_precision_score(y_true, m_scores))
            m_auroc, m_se = auroc_delong_se(y_true, m_scores)
            # 95% CI on AUROC
            auroc_lo = m_auroc - 1.96 * m_se
            auroc_hi = m_auroc + 1.96 * m_se
            below = auroc_hi < 0.5
            auroc_rows.append({
                "tag": args.tag, "gold_standard": gs_name, "candidate_set": args.candidate_set,
                "method": method_name, "auroc": m_auroc, "auroc_se": m_se,
                "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
                "below_random_auroc": bool(below), "aupr": m_aupr,
            })
            # Paired bootstrap on AUPR
            dm, lo, hi = paired_aupr_bootstrap(y_true, m_scores, r_scores, B=args.B, sub_n=args.sub_n, seed=args.seed)
            boot_rows.append({
                "tag": args.tag, "gold_standard": gs_name, "candidate_set": args.candidate_set,
                "method": method_name,
                "aupr_obs": m_aupr, "aupr_random": r_aupr,
                "delta_mean": dm, "delta_ci_lo": lo, "delta_ci_hi": hi,
                "significantly_below": bool(hi < 0),
            })
            print(f"  {method_name}: AUROC={m_auroc:.4f}±{m_se:.4f} (CI {auroc_lo:.4f}-{auroc_hi:.4f} below={below}) "
                  f"AUPR={m_aupr:.4g} Δ={dm:.4g} CI=[{lo:.4g}, {hi:.4g}] ({time.time()-t0:.1f}s)", flush=True)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(auroc_rows).to_parquet(EVAL_ROOT / f"{args.tag}_auroc_ci.parquet", index=False)
    pd.DataFrame(boot_rows).to_parquet(EVAL_ROOT / f"{args.tag}_paired_bootstrap.parquet", index=False)
    print(f"\n[{args.tag}] wrote auroc_ci + paired_bootstrap")


if __name__ == "__main__":
    main()
