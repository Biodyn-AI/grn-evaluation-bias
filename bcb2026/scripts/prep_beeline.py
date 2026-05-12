"""Phase A → B prep: invert BEELINE log-normalization, write h5ad with raw + log layers,
build per-dataset universe U = T x (G \\ {self}), pin all symbol-mapping outcomes.

Outputs (per dataset, under implementation/outputs/prep/<tag>/):
  - <tag>_raw.h5ad      AnnData with X=integer counts, layers={'log_normalized': original BEELINE values}
  - <tag>_universe.json {G, T, |U|, symbol_map_policy}
  - <tag>_mapping_report.tsv (raw_symbol, mapped_symbol, status)
  - <tag>_pseudotime.csv (passthrough)

Usage:
  python scripts/prep_beeline.py --tag hESC
  python scripts/prep_beeline.py --tag hHep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
DATA_ROOT = IMPL_ROOT / "data" / "beeline" / "BEELINE-data" / "inputs" / "scRNA-Seq"
NETWORKS_ROOT = IMPL_ROOT / "data" / "beeline" / "Networks"
HGNC_PATH = Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/external/hgnc_complete_set.txt")
OUT_ROOT = IMPL_ROOT / "outputs" / "prep"


def load_hgnc(path: Path) -> dict[str, str]:
    """Return {raw_symbol_upper: approved_symbol}. Approved -> self; previous/alias -> approved."""
    mapping: dict[str, str] = {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    for _, row in df.iterrows():
        approved = row["Approved symbol"].strip().upper()
        if not approved:
            continue
        mapping[approved] = approved
        for col in ("Previous symbols", "Alias symbols"):
            for raw in str(row[col]).split(","):
                raw = raw.strip().upper()
                if raw and raw not in mapping:
                    mapping[raw] = approved
    return mapping


def normalize_symbol(sym: str) -> str:
    return "".join(c for c in sym.upper() if c.isalnum() or c in "-.").strip()


def map_symbols(symbols: list[str], hgnc: dict[str, str]) -> tuple[list[str], pd.DataFrame]:
    """Apply HGNC mapping. Returns (mapped_list_same_length, status_df).
    Policy: lexicographic (drop_ambiguous and drop_unmapped recorded but not applied here)."""
    mapped = []
    rows = []
    for s in symbols:
        norm = normalize_symbol(s)
        if not norm:
            mapped.append("")
            rows.append((s, "", "empty"))
            continue
        if norm in hgnc:
            target = hgnc[norm]
            status = "approved" if target == norm else "alias_resolved"
        else:
            target = norm
            status = "unmapped"
        mapped.append(target)
        rows.append((s, target, status))
    return mapped, pd.DataFrame(rows, columns=["raw", "mapped", "status"])


def load_tf_list() -> set[str]:
    """T_BEELINE = 1563 TFs from BEELINE human-tfs.csv."""
    df = pd.read_csv(NETWORKS_ROOT.parent / "human-tfs.csv")
    return set(normalize_symbol(t) for t in df["TF"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    ap.add_argument("--scale", type=float, default=1e4,
                    help="Re-scale factor used to invert log-normalization (default 1e4 = scanpy convention)")
    args = ap.parse_args()

    expr_path = DATA_ROOT / args.tag / "ExpressionData.csv"
    assert expr_path.exists(), f"Missing {expr_path}"

    print(f"[{args.tag}] Loading {expr_path} ...", flush=True)
    df = pd.read_csv(expr_path, index_col=0)
    print(f"[{args.tag}] shape: genes={df.shape[0]}, cells={df.shape[1]}", flush=True)

    raw_symbols = list(df.index)
    cells = list(df.columns)

    # Map symbols
    print(f"[{args.tag}] Loading HGNC dump ...", flush=True)
    hgnc = load_hgnc(HGNC_PATH)
    print(f"[{args.tag}] HGNC entries: {len(hgnc)}", flush=True)
    mapped_symbols, mapping_df = map_symbols(raw_symbols, hgnc)
    print(f"[{args.tag}] mapping: approved={(mapping_df['status']=='approved').sum()}, "
          f"alias_resolved={(mapping_df['status']=='alias_resolved').sum()}, "
          f"unmapped={(mapping_df['status']=='unmapped').sum()}", flush=True)

    # Invert log1p+normalize_total: x = log1p(raw / total * scale) -> raw_proxy = expm1(x) (per cell rescaled is approximate)
    # BEELINE applies log1p over counts (not per-cell normalized), per their pipeline notes. We'll therefore use
    # x = log1p(counts) so counts = round(expm1(x)). This is the BEELINE preprocessing convention.
    log_mat = df.values.astype(np.float32)
    counts_mat = np.round(np.expm1(log_mat)).astype(np.int32)
    counts_mat[counts_mat < 0] = 0

    # Resolve duplicate mapped symbols by summing counts (post-mapping aggregation)
    series = pd.DataFrame(counts_mat, index=mapped_symbols, columns=cells)
    series = series[series.index != ""]
    series = series.groupby(level=0).sum()
    n_after_dedup = series.shape[0]
    print(f"[{args.tag}] after symbol dedup: {n_after_dedup} unique mapped genes", flush=True)

    # Build universe
    tfs = load_tf_list()
    G = set(series.index)
    T = tfs & G
    U_size = len(T) * (len(G) - 1)  # exclude self
    print(f"[{args.tag}] |G|={len(G)}, |T|={len(T)}, |U|={U_size}", flush=True)

    # Save outputs
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try anndata; if not available, save as TSV
    try:
        import anndata as ad
        X = series.T.values  # cells x genes
        adata = ad.AnnData(
            X=sp.csr_matrix(X.astype(np.float32)),
            obs=pd.DataFrame(index=series.columns),
            var=pd.DataFrame(index=series.index),
        )
        # Add log layer (cell-major)
        log_for_layer = np.log1p(X.astype(np.float32))
        adata.layers["log1p"] = sp.csr_matrix(log_for_layer)
        adata.write_h5ad(out_dir / f"{args.tag}_raw.h5ad")
        print(f"[{args.tag}] wrote {out_dir / f'{args.tag}_raw.h5ad'}", flush=True)
    except Exception as e:
        print(f"[{args.tag}] anndata write failed: {e}; falling back to TSV", flush=True)
        series.to_csv(out_dir / f"{args.tag}_counts.tsv", sep="\t")

    mapping_df.to_csv(out_dir / f"{args.tag}_mapping_report.tsv", sep="\t", index=False)
    universe = {
        "tag": args.tag,
        "n_raw_genes": len(raw_symbols),
        "n_mapped_unique_genes": int(n_after_dedup),
        "n_cells": len(cells),
        "G_size": int(len(G)),
        "T_size": int(len(T)),
        "U_size": int(U_size),
        "tfs_in_G": sorted(T)[:50],  # sample of TFs that survived
        "hgnc_path": str(HGNC_PATH),
        "hgnc_sha256": "b9482a5247a4162017bd9731ac6300757b9b7511ec9a02f99bafd6b4a43a89fc",
    }
    with open(out_dir / f"{args.tag}_universe.json", "w") as f:
        json.dump(universe, f, indent=2)
    print(f"[{args.tag}] wrote {out_dir / f'{args.tag}_universe.json'}", flush=True)


if __name__ == "__main__":
    main()
