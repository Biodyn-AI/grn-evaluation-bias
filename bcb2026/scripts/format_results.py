"""Phase C helper: generate LaTeX-ready snippets from eval outputs.

Reads outputs/eval/*.parquet and prints LaTeX table fragments + headline
numbers for the paper. Numbers are quoted verbatim from CSVs (no rounding from memory)
to satisfy the cross-check requirement (Phase D).

Outputs:
  outputs/eval/paper_snippets.md   markdown with all numbers + the LaTeX snippet for each.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent / "outputs" / "eval"
PREP_ROOT = HERE.parent / "outputs" / "prep"


def load_all_metrics() -> pd.DataFrame:
    frames = []
    for p in EVAL_ROOT.glob("*_metrics.parquet"):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("No *_metrics.parquet in outputs/eval")
    return pd.concat(frames, ignore_index=True)


def fmt_universe_row(tag: str) -> str:
    with open(PREP_ROOT / tag / f"{tag}_universe.json") as f:
        uni = json.load(f)
    return (f"{tag} & {uni['G_size']:,} & {uni['T_size']:,} & {uni['U_size']:,} "
            f"& \\texttt{{TBA}} & \\texttt{{TBA}}")


def main():
    out_lines = []
    out_lines.append("# Paper snippets (auto-generated from outputs/eval)\n")
    out_lines.append("## Universe table\n")
    for tag in ("hESC", "hHep"):
        out_lines.append(fmt_universe_row(tag) + " \\\\")

    if not any(EVAL_ROOT.glob("*_metrics.parquet")):
        out_lines.append("\n(no metrics yet; rerun after eval_universe_aware finishes)")
    else:
        df = load_all_metrics()
        out_lines.append("\n## Median AUPR by method, candidate set\n")
        piv = df.groupby(["tag", "candidate_set", "method"])["aupr"].median().unstack("method")
        out_lines.append("```\n" + piv.to_string() + "\n```\n")
        out_lines.append("\n## Median AUROC by method, candidate set\n")
        piv2 = df.groupby(["tag", "candidate_set", "method"])["auroc"].median().unstack("method")
        out_lines.append("```\n" + piv2.to_string() + "\n```\n")
        # Candidate-set inflation factor for scGPT
        out_lines.append("\n## Candidate-set AUPR inflation per dataset (median across gold standards)\n")
        for tag in ("hESC", "hHep"):
            sub = df[df["tag"] == tag]
            piv = sub.groupby(["method", "candidate_set"])["aupr"].median().unstack("candidate_set")
            if {"all_pairs","tf_sources_targets"}.issubset(piv.columns):
                piv["inflation_tf_sources_targets_vs_all_pairs"] = piv["tf_sources_targets"] / piv["all_pairs"]
            out_lines.append(f"### {tag}\n```\n" + piv.to_string() + "\n```")

    perm_files = list(EVAL_ROOT.glob("*_permutation.parquet"))
    if perm_files:
        out_lines.append("\n## Permutation null + below-random verdicts\n")
        perm = pd.concat([pd.read_parquet(p) for p in perm_files], ignore_index=True)
        out_lines.append("```\n" + perm.to_string() + "\n```\n")
    boot_files = list(EVAL_ROOT.glob("*_paired_bootstrap.parquet"))
    if boot_files:
        out_lines.append("\n## Paired bootstrap (model - random) 95% CI\n")
        boot = pd.concat([pd.read_parquet(b) for b in boot_files], ignore_index=True)
        out_lines.append("```\n" + boot.to_string() + "\n```\n")

    aov_path = EVAL_ROOT / "variance_decomp_log_aupr.csv"
    if aov_path.exists():
        out_lines.append("\n## Variance decomposition (log10 AUPR)\n")
        out_lines.append("```\n" + aov_path.read_text() + "\n```\n")

    out_path = EVAL_ROOT / "paper_snippets.md"
    out_path.write_text("\n".join(out_lines))
    print(f"wrote {out_path}")
    print("\n--- Preview ---")
    print("\n".join(out_lines[:60]))


if __name__ == "__main__":
    main()
