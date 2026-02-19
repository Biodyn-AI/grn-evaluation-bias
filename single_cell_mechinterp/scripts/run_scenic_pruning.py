from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from ctxcore.rnkdb import FeatherRankingDatabase
from pyscenic.prune import prune2df
from pyscenic.utils import modules_from_adjacencies


def _sample_cells(adata: sc.AnnData, max_cells: int | None, seed: int) -> sc.AnnData:
    if not max_cells or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
    return adata[indices].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCENIC motif pruning on GRNBoost2 adjacencies.")
    parser.add_argument(
        "--adjacencies",
        default="outputs/eval_bias/baselines_immune_subset/immune_subset_hpn/scenic_grnboost2_edges.tsv",
    )
    parser.add_argument(
        "--expression-h5ad",
        default="outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad",
    )
    parser.add_argument(
        "--ranking-db",
        default="external/scenic/hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.genes_vs_motifs.rankings.feather",
    )
    parser.add_argument(
        "--motif-annotations",
        default="external/scenic/motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
    )
    parser.add_argument(
        "--output",
        default="outputs/eval_bias/baselines_immune_subset/immune_subset_hpn/scenic_pruned_edges.tsv",
    )
    parser.add_argument("--max-cells", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-genes", type=int, default=10)
    args = parser.parse_args()

    adj_path = Path(args.adjacencies)
    expr_path = Path(args.expression_h5ad)
    output_path = Path(args.output)

    adj = pd.read_csv(adj_path, sep="\t")
    adj = adj.rename(columns={"source": "TF", "target": "target", "score": "importance"})
    adj = adj[["TF", "target", "importance"]].dropna()

    gene_set = sorted(set(adj["TF"]) | set(adj["target"]))

    adata = sc.read_h5ad(expr_path)
    adata = adata[:, adata.var_names.isin(gene_set)].copy()
    adata = _sample_cells(adata, args.max_cells, args.seed)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr_df = pd.DataFrame(X, columns=adata.var_names)

    modules = modules_from_adjacencies(
        adj,
        expr_df,
        min_genes=args.min_genes,
    )

    ranking_db = FeatherRankingDatabase(str(args.ranking_db), name="hg38_refseq")
    pruned = prune2df(
        [ranking_db],
        modules,
        str(args.motif_annotations),
        client_or_address="custom_multiprocessing",
        num_workers=1,
        module_chunksize=50,
    )

    if pruned.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pruned.to_csv(output_path, sep="\t", index=False)
        print("No pruned edges found; wrote empty file", output_path)
        return

    if isinstance(pruned.columns, pd.MultiIndex):
        pruned.columns = [
            "_".join([str(part) for part in col if part and str(part) != "nan"]) for col in pruned.columns
        ]

    source_col = next((col for col in pruned.columns if col.lower() in {"tf", "source"}), None)
    target_col = next((col for col in pruned.columns if col.lower() in {"target"}), None)
    if (source_col is None or target_col is None) and isinstance(pruned.index, pd.MultiIndex):
        pruned = pruned.reset_index()
        source_col = next((col for col in pruned.columns if col.lower() in {"tf", "source"}), None)
        target_col = next((col for col in pruned.columns if col.lower() in {"target"}), None)
    score_candidates = [
        col
        for col in pruned.columns
        if col.lower() in {"nes", "enrichment", "auc"} or col.lower().endswith("nes")
    ]
    score_col = score_candidates[0] if score_candidates else pruned.columns[-1]

    if source_col is None:
        raise ValueError(f"Unable to locate source column in pruned output: {pruned.columns}")

    if target_col is None and "Enrichment_TargetGenes" in pruned.columns:
        def _split_targets(value: object) -> list[str]:
            if isinstance(value, (list, tuple, set, np.ndarray)):
                items = []
                for item in value:
                    if isinstance(item, (list, tuple)) and item:
                        items.append(str(item[0]))
                    else:
                        items.append(str(item))
                return [item for item in items if item]
            if not isinstance(value, str):
                return []
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            text = text.replace(";", ",")
            parts = [part.strip() for part in text.split(",") if part.strip()]
            targets = []
            for part in parts:
                cleaned = part.strip().strip("'\"").strip("()")
                if not cleaned:
                    continue
                if " " in cleaned:
                    cleaned = cleaned.split(" ")[0]
                targets.append(cleaned)
            return targets

        rows = []
        for _, row in pruned.iterrows():
            targets = _split_targets(row.get("Enrichment_TargetGenes"))
            for target in targets:
                rows.append(
                    {
                        "source": row[source_col],
                        "target": target,
                        "score": row[score_col],
                    }
                )
        pruned = pd.DataFrame(rows)
    elif target_col is not None:
        pruned = pruned[[source_col, target_col, score_col]].rename(
            columns={source_col: "source", target_col: "target", score_col: "score"}
        )
    else:
        raise ValueError(
            f"Unable to locate target genes in pruned output: {pruned.columns}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(output_path, sep="\t", index=False)
    print("Saved pruned edges to", output_path)


if __name__ == "__main__":
    main()
