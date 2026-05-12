"""Phase B5: permutation null + paired bootstrap for the 'below random' claim.

For each (dataset, gold_standard, candidate_set, method):
  - Compute observed AUPR.
  - Generate K (default 1000) permutation nulls by shuffling scores over the candidate set.
  - Empirical p-value = (#perm_AUPR >= obs + 1) / (K+1).
  - Below-random decision: obs < 5th-percentile of null distribution.
  - Paired bootstrap of (model_AUPR - random_AUPR): resample candidate-set rows with replacement
    B (default 2000) times, recompute both AUPRs, report mean and 95% percentile CI of the diff.

Outputs:
  outputs/eval/<tag>_permutation.parquet
  outputs/eval/<tag>_paired_bootstrap.parquet

Usage:
  python scripts/permutation_null.py --tag hHep --n-perm 1000 --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
EVAL_ROOT = IMPL_ROOT / "outputs" / "eval"
PREP_ROOT = IMPL_ROOT / "outputs" / "prep"
BASELINES_ROOT = IMPL_ROOT / "outputs" / "baselines"
SCGPT_ROOT = IMPL_ROOT / "outputs" / "scgpt"
GENEFORMER_ROOT = IMPL_ROOT / "outputs" / "geneformer"

# Reuse the gold-standard config from eval_universe_aware
import sys
sys.path.insert(0, str(HERE))
from eval_universe_aware import (
    GOLD_STANDARDS, load_gold, load_universe, method_files, normalize_symbol,
    build_candidate_mask,
)


def evaluate_aupr(method_df_sub: pd.DataFrame, gold_pairs: set[tuple[str, str]]) -> tuple[float, np.ndarray, np.ndarray]:
    src = method_df_sub["source"].values
    tgt = method_df_sub["target"].values
    scores = np.nan_to_num(method_df_sub["score"].values.astype(np.float32))
    y_true = np.fromiter(((s, t) in gold_pairs for s, t in zip(src, tgt)), dtype=bool, count=len(src))
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, scores, y_true
    return float(average_precision_score(y_true, scores)), scores, y_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    uni, genes, T = load_universe(args.tag)
    genes_set, T_set = set(genes), set(T)
    methods = method_files(args.tag)
    print(f"[{args.tag}] universe |G|={len(genes_set)} |T|={len(T_set)}; methods={sorted(methods.keys())}", flush=True)

    # Need a "random" baseline file to do paired bootstrap; use the parquet from B3
    random_path = BASELINES_ROOT / args.tag / "random.parquet"
    if not random_path.exists():
        raise SystemExit(f"Need random baseline at {random_path}")
    random_df = pd.read_parquet(random_path)
    random_df["source"] = random_df["source"].map(normalize_symbol)
    random_df["target"] = random_df["target"].map(normalize_symbol)

    rng = np.random.default_rng(args.seed)
    perm_rows = []
    boot_rows = []
    for gs_name, gs_path in GOLD_STANDARDS[args.tag].items():
        gold = load_gold(gs_path, genes_set)
        gold_pairs = set(zip(gold["source"], gold["target"]))
        gold_tfs, gold_targets = set(gold["source"]), set(gold["target"])
        for candidate_set in ("all_pairs", "tf_sources", "tf_sources_targets"):
            # Random baseline subset (the same across methods, used for paired bootstrap)
            rmask = build_candidate_mask(random_df, candidate_set, T_set, gold_tfs, gold_targets, genes_set)
            rsub = random_df.loc[rmask].reset_index(drop=True)
            r_aupr, r_scores, r_y = evaluate_aupr(rsub, gold_pairs)
            print(f"\n[{args.tag}/{gs_name}/{candidate_set}] random AUPR={r_aupr:.4g}, n_pos={int(r_y.sum())}/{len(rsub)}", flush=True)

            for method_name, method_path in methods.items():
                df = pd.read_parquet(method_path)
                df["source"] = df["source"].map(normalize_symbol)
                df["target"] = df["target"].map(normalize_symbol)
                df = df[df["source"].isin(genes_set) & df["target"].isin(genes_set)]
                df = df[df["source"] != df["target"]]
                mmask = build_candidate_mask(df, candidate_set, T_set, gold_tfs, gold_targets, genes_set)
                msub = df.loc[mmask].reset_index(drop=True)
                # Align to same row order as random sub (by source,target) — both share universe
                msub_idx = msub.set_index(["source", "target"]).reindex(
                    list(zip(rsub["source"], rsub["target"]))
                )
                m_aligned_scores = np.nan_to_num(msub_idx["score"].values.astype(np.float32))
                aligned = pd.DataFrame({"source": rsub["source"], "target": rsub["target"], "score": m_aligned_scores})
                obs_aupr, obs_scores, obs_y = evaluate_aupr(aligned, gold_pairs)
                if np.isnan(obs_aupr) or np.isnan(r_aupr):
                    continue
                # Permutation null on scores
                t0 = time.time()
                null = np.empty(args.n_perm, dtype=np.float32)
                idx = np.arange(len(obs_scores))
                for k in range(args.n_perm):
                    perm = rng.permutation(idx)
                    null[k] = average_precision_score(obs_y, obs_scores[perm])
                p_value = (np.sum(null >= obs_aupr) + 1) / (args.n_perm + 1)
                below_random = obs_aupr < np.percentile(null, 5)
                perm_rows.append({
                    "tag": args.tag, "gold_standard": gs_name, "candidate_set": candidate_set,
                    "method": method_name, "obs_aupr": obs_aupr,
                    "null_mean": float(null.mean()), "null_q5": float(np.percentile(null, 5)),
                    "null_q95": float(np.percentile(null, 95)),
                    "p_value": float(p_value), "below_random": bool(below_random),
                })

                # Paired bootstrap of (obs - random)
                diffs = np.empty(args.n_boot, dtype=np.float32)
                for k in range(args.n_boot):
                    bidx = rng.integers(0, len(obs_scores), size=len(obs_scores))
                    if obs_y[bidx].sum() == 0 or obs_y[bidx].sum() == len(bidx):
                        diffs[k] = np.nan
                        continue
                    m = average_precision_score(obs_y[bidx], obs_scores[bidx])
                    r = average_precision_score(obs_y[bidx], r_scores[bidx])
                    diffs[k] = m - r
                ok = ~np.isnan(diffs)
                if ok.sum() > 0:
                    boot_rows.append({
                        "tag": args.tag, "gold_standard": gs_name, "candidate_set": candidate_set,
                        "method": method_name,
                        "obs_aupr": obs_aupr, "random_aupr": r_aupr,
                        "diff_mean": float(np.mean(diffs[ok])),
                        "diff_ci_lo": float(np.percentile(diffs[ok], 2.5)),
                        "diff_ci_hi": float(np.percentile(diffs[ok], 97.5)),
                        "significant_below": bool(np.percentile(diffs[ok], 97.5) < 0),
                    })
                print(f"  {method_name}: AUPR={obs_aupr:.4g} null_mean={null.mean():.4g} "
                      f"p={p_value:.3g} below_rand={below_random} elapsed={time.time()-t0:.0f}s", flush=True)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perm_rows).to_parquet(EVAL_ROOT / f"{args.tag}_permutation.parquet", index=False)
    pd.DataFrame(boot_rows).to_parquet(EVAL_ROOT / f"{args.tag}_paired_bootstrap.parquet", index=False)
    print(f"\n[{args.tag}] wrote permutation + bootstrap outputs", flush=True)


if __name__ == "__main__":
    main()
