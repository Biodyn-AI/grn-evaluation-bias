#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network_inference.src.data.mapping import load_hgnc_alias_map_extended, map_edge_table
from network_inference.src.utils.scm_imports import ensure_mechinterp_path

ensure_mechinterp_path()
from src.eval.gene_symbols import normalize_edges, normalize_gene_names
from network_inference.src.evaluation.metrics import aupr


def load_var_names(h5ad_path: Path) -> List[str]:
    with h5py.File(h5ad_path, "r") as handle:
        if "var" not in handle:
            raise KeyError("Missing var group in h5ad file")
        var = handle["var"]
        for key in ("_index", "index", "var_names"):
            if key in var:
                data = var[key][()]
                return _decode_h5ad_strings(data)
    raise KeyError("Unable to locate var index in h5ad file")


def _decode_h5ad_strings(values: np.ndarray) -> List[str]:
    if values.dtype.kind in ("S", "O"):
        return [
            value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            for value in values
        ]
    return [str(value) for value in values]


def read_edges(path: Path, score_col: str | None = "score") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["source", "target"]
    if score_col and score_col in df.columns:
        required.append(score_col)
    df = df[required].dropna().copy()
    df.columns = ["source", "target"] + (["score"] if len(required) == 3 else [])
    if "score" not in df.columns:
        df["score"] = 1.0
    return df


def normalize_edges_table(
    edges: pd.DataFrame,
    alias_map: Dict[str, str],
    mapping_cfg: Dict[str, object] | None,
    gene_set: set[str],
) -> pd.DataFrame:
    edges = map_edge_table(edges, mapping_cfg, alias_map)
    edges = normalize_edges(edges, alias_map)
    edges = edges[edges["source"].isin(gene_set) & edges["target"].isin(gene_set)]
    if "score" in edges.columns:
        edges = edges.groupby(["source", "target"], as_index=False)["score"].max()
    else:
        edges = edges.drop_duplicates()
    return edges


def build_truth_mask(
    truth_edges: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    gene_count: int,
) -> np.ndarray:
    truth_mask = np.zeros((gene_count, gene_count), dtype=bool)
    for _, row in truth_edges.iterrows():
        src = gene_to_idx.get(row["source"])
        tgt = gene_to_idx.get(row["target"])
        if src is None or tgt is None:
            continue
        truth_mask[src, tgt] = True
    return truth_mask


def score_aupr(
    pred_edges: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    gene_count: int,
    truth_mask: np.ndarray,
) -> float:
    scores = np.zeros((gene_count, gene_count), dtype=np.float32)
    for _, row in pred_edges.iterrows():
        src = gene_to_idx.get(row["source"])
        tgt = gene_to_idx.get(row["target"])
        if src is None or tgt is None:
            continue
        score = float(row["score"]) if "score" in row else 1.0
        if score > scores[src, tgt]:
            scores[src, tgt] = score
    scores_flat = scores.ravel()
    labels_flat = truth_mask.ravel().astype(np.int8)
    return float(aupr(scores_flat, labels_flat))


def load_csv_table(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate time-series summary metrics CSV")
    parser.add_argument(
        "--output",
        default="network_inference/outputs/summary_timeseries_metrics.csv",
        help="Path to write the summary CSV",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent
    outputs_dir = root / "outputs"
    data_dir = root / "data"

    h5ad_path = repo_root / "single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad"
    hgnc_alias = repo_root / "single_cell_mechinterp/external/hgnc_complete_set.txt"

    hpn_mapping = {
        "strip_phospho_suffix": True,
        "symbol_map_tsv": data_dir / "hpn_dream_symbol_map.tsv",
        "hgnc_extra_alias_cols": ["entrez_id", "ensembl_id", "uniprot_ids"],
        "drop_unmapped": True,
    }
    beeline_mapping = {
        "symbol_map_tsv": data_dir / "beeline_gsd_symbol_map.tsv",
        "hgnc_extra_alias_cols": ["entrez_id", "ensembl_id", "uniprot_ids"],
        "drop_unmapped": True,
    }

    var_names = load_var_names(h5ad_path)
    alias_map = load_hgnc_alias_map_extended(hgnc_alias, hpn_mapping["hgnc_extra_alias_cols"])
    gene_names_norm = normalize_gene_names(np.array(var_names, dtype=object), alias_map)
    gene_set = set(gene_names_norm)
    gene_count = len(gene_names_norm)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_norm)}
    hpn_truth = read_edges(data_dir / "hpn_dream_goldstandard.tsv", score_col=None)
    hpn_truth = normalize_edges_table(hpn_truth, alias_map, hpn_mapping, gene_set)
    hpn_truth_mask = build_truth_mask(hpn_truth, gene_to_idx, gene_count)
    np.fill_diagonal(hpn_truth_mask, False)

    beeline_truth = read_edges(data_dir / "beeline_gsd_refnetwork.tsv", score_col=None)
    beeline_truth = normalize_edges_table(beeline_truth, alias_map, beeline_mapping, gene_set)
    beeline_truth_mask = build_truth_mask(beeline_truth, gene_to_idx, gene_count)
    np.fill_diagonal(beeline_truth_mask, False)

    cross_eval_lookup: Dict[str, Tuple[float, float]] = {}
    cross_eval_lookup["omnipath_dorothea_intersection_immune"] = (
        load_json(outputs_dir / "cross_eval_dorothea_intersection_immune_hpn.json")["hpn"]["aupr"],
        load_json(outputs_dir / "cross_eval_dorothea_intersection_immune_hpn.json")["beeline"]["aupr"],
    )
    cross_eval_lookup["omnipath_dorothea_union_immune"] = (
        load_json(outputs_dir / "cross_eval_dorothea_union_immune_hpn.json")["hpn"]["raw"]["aupr"],
        load_json(outputs_dir / "cross_eval_dorothea_union_immune_hpn.json")["beeline"]["raw"]["aupr"],
    )
    cross_eval_lookup["omnipath_dorothea_intersection_scaled"] = (
        load_json(outputs_dir / "cross_eval_dorothea_intersection_immune_hpn_scaled.json")["hpn"]["aupr"],
        load_json(outputs_dir / "cross_eval_dorothea_intersection_immune_hpn_scaled.json")["beeline"]["aupr"],
    )
    cross_eval_lookup["omnipath_dorothea_union_scaled"] = (
        load_json(outputs_dir / "cross_eval_dorothea_union_immune_hpn_scaled.json")["hpn"]["aupr"],
        load_json(outputs_dir / "cross_eval_dorothea_union_immune_hpn_scaled.json")["beeline"]["aupr"],
    )
    union_1200 = load_json(outputs_dir / "cross_eval_dorothea_union_immune_hpn_scaled_1200x420.json")
    cross_eval_lookup["omnipath_dorothea_union_scaled_1200x420"] = (
        union_1200["hpn_aupr"],
        union_1200["beeline_aupr"],
    )
    intercell = load_json(outputs_dir / "cross_eval_intercell_union_variants_immune_hpn.json")
    cross_eval_lookup["omnipath_intercell_union"] = (
        intercell["intercell_union"]["hpn"]["aupr"],
        intercell["intercell_union"]["beeline"]["aupr"],
    )
    cross_eval_lookup["omnipath_intercell_union_sources"] = (
        intercell["intercell_union_sources"]["hpn"]["aupr"],
        intercell["intercell_union_sources"]["beeline"]["aupr"],
    )
    cross_eval_lookup["omnipath_intercell_union_targets"] = (
        intercell["intercell_union_targets"]["hpn"]["aupr"],
        intercell["intercell_union_targets"]["beeline"]["aupr"],
    )
    cross_eval_lookup["omnipath_intercell_union_sources_relaxed"] = (
        intercell["intercell_union_sources_relaxed"]["hpn"]["aupr"],
        intercell["intercell_union_sources_relaxed"]["beeline"]["aupr"],
    )

    learned_methods = {
        "omnipath_relaxed_immune": {
            "edges": outputs_dir / "inferred_edges_omnipath_relaxed_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd.json",
        },
        "omnipath_dorothea_intersection_immune": {
            "edges": outputs_dir / "inferred_edges_omnipath_dorothea_intersection_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_dorothea_intersection_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_dorothea_intersection_immune.json",
        },
        "omnipath_dorothea_union_immune": {
            "edges": outputs_dir / "inferred_edges_omnipath_dorothea_union_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_dorothea_union_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_dorothea_union_immune.json",
        },
        "omnipath_intercell_union": {
            "edges": outputs_dir / "inferred_edges_omnipath_intercell_union_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_intercell_union_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_intercell_union_immune.json",
        },
        "omnipath_intercell_union_sources": {
            "edges": outputs_dir / "inferred_edges_omnipath_intercell_union_sources_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_intercell_union_sources_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_intercell_union_sources_immune.json",
        },
        "omnipath_intercell_union_targets": {
            "edges": outputs_dir / "inferred_edges_omnipath_intercell_union_targets_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_intercell_union_targets_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_intercell_union_targets_immune.json",
        },
        "omnipath_intercell_union_sources_relaxed": {
            "edges": outputs_dir / "inferred_edges_omnipath_intercell_union_sources_relaxed_immune_hpn.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_intercell_union_sources_relaxed_immune.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_intercell_union_sources_relaxed_immune.json",
        },
        "omnipath_dorothea_intersection_scaled": {
            "edges": outputs_dir / "inferred_edges_omnipath_dorothea_intersection_immune_hpn_scaled.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_dorothea_intersection_immune_scaled.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_dorothea_intersection_immune_scaled.json",
        },
        "omnipath_dorothea_union_scaled": {
            "edges": outputs_dir / "inferred_edges_omnipath_dorothea_union_immune_hpn_scaled.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_dorothea_union_immune_scaled.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_dorothea_union_immune_scaled.json",
        },
        "omnipath_dorothea_union_scaled_1200x420": {
            "edges": outputs_dir / "inferred_edges_omnipath_dorothea_union_immune_hpn_scaled_1200x420.tsv",
            "hpn": outputs_dir / "timeseries_eval_hpn_dream_dorothea_union_immune_scaled_1200x420.json",
            "beeline": outputs_dir / "timeseries_eval_beeline_gsd_dorothea_union_immune_scaled_1200x420.json",
        },
    }

    baseline_priors = load_csv_table(outputs_dir / "baseline_eval_hpn_beeline.csv")
    baseline_priors = {row["method"]: row for row in baseline_priors}

    grn_baselines = load_csv_table(outputs_dir / "grn_baseline_eval_hpn_beeline.csv")
    grn_baselines = {row["method"]: row for row in grn_baselines}

    score_eval = load_csv_table(outputs_dir / "score_eval_grn_baselines_immune.csv")
    score_eval_lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in score_eval:
        score_eval_lookup[(row["method"], row["reference"])] = row

    order = [
        "omnipath_relaxed_immune",
        "omnipath_dorothea_intersection_immune",
        "omnipath_dorothea_union_immune",
        "omnipath_intercell_union",
        "omnipath_intercell_union_sources",
        "omnipath_intercell_union_targets",
        "omnipath_intercell_union_sources_relaxed",
        "omnipath_dorothea_intersection_scaled",
        "omnipath_dorothea_union_scaled",
        "omnipath_prior",
        "trrust_prior",
        "random_3950",
        "genie3",
        "grnboost2",
        "pearson",
        "spearman",
        "scenic_grnboost2",
        "pidc_proxy",
        "omnipath_dorothea_union_scaled_1200x420",
        "scenic_pruned",
        "pidc_full",
        "random",
    ]

    rows = []
    for method in order:
        if method in learned_methods:
            entry = learned_methods[method]
            hpn_metrics = load_json(entry["hpn"])
            beeline_metrics = load_json(entry["beeline"])

            edges_path = entry["edges"]
            edges_raw = read_edges(edges_path)
            if method in cross_eval_lookup:
                hpn_aupr, beeline_aupr = cross_eval_lookup[method]
            else:
                edges_norm = normalize_edges_table(edges_raw, alias_map, hpn_mapping, gene_set)
                hpn_aupr = score_aupr(edges_norm, gene_to_idx, gene_count, hpn_truth_mask)
                beeline_edges_norm = normalize_edges_table(
                    edges_raw,
                    alias_map,
                    beeline_mapping,
                    gene_set,
                )
                beeline_aupr = score_aupr(
                    beeline_edges_norm,
                    gene_to_idx,
                    gene_count,
                    beeline_truth_mask,
                )

            rows.append(
                {
                    "method": method,
                    "edges": int(edges_raw.shape[0]),
                    "hpn_precision": hpn_metrics["precision"],
                    "hpn_recall": hpn_metrics["recall"],
                    "hpn_f1": hpn_metrics["f1"],
                    "hpn_aupr": hpn_aupr,
                    "beeline_precision": beeline_metrics["precision"],
                    "beeline_recall": beeline_metrics["recall"],
                    "beeline_f1": beeline_metrics["f1"],
                    "beeline_aupr": beeline_aupr,
                }
            )
            continue

        if method in baseline_priors:
            baseline = baseline_priors[method]
            edges_path = outputs_dir / f"baseline_edges_{method.replace('_prior', '')}.tsv"
            if method == "random_3950":
                edges_path = outputs_dir / "baseline_edges_random_3950.tsv"
            edges_raw = read_edges(edges_path, score_col=None)
            edges_norm = normalize_edges_table(edges_raw, alias_map, hpn_mapping, gene_set)
            hpn_aupr = score_aupr(edges_norm, gene_to_idx, gene_count, hpn_truth_mask)
            beeline_edges_norm = normalize_edges_table(edges_raw, alias_map, beeline_mapping, gene_set)
            beeline_aupr = score_aupr(
                beeline_edges_norm,
                gene_to_idx,
                gene_count,
                beeline_truth_mask,
            )

            rows.append(
                {
                    "method": method,
                    "edges": int(baseline["edges"]),
                    "hpn_precision": float(baseline["hpn_precision"]),
                    "hpn_recall": float(baseline["hpn_recall"]),
                    "hpn_f1": float(baseline["hpn_f1"]),
                    "hpn_aupr": hpn_aupr,
                    "beeline_precision": float(baseline["beeline_precision"]),
                    "beeline_recall": float(baseline["beeline_recall"]),
                    "beeline_f1": float(baseline["beeline_f1"]),
                    "beeline_aupr": beeline_aupr,
                }
            )
            continue

        if method in grn_baselines:
            baseline = grn_baselines[method]
            hpn_score = score_eval_lookup.get((method, "hpn_dream"))
            beeline_score = score_eval_lookup.get((method, "beeline_gsd"))
            rows.append(
                {
                    "method": method,
                    "edges": int(float(baseline["edges"])),
                    "hpn_precision": float(baseline["hpn_precision"]),
                    "hpn_recall": float(baseline["hpn_recall"]),
                    "hpn_f1": float(baseline["hpn_f1"]),
                    "hpn_aupr": float(hpn_score["aupr"]) if hpn_score else 0.0,
                    "beeline_precision": float(baseline["beeline_precision"]),
                    "beeline_recall": float(baseline["beeline_recall"]),
                    "beeline_f1": float(baseline["beeline_f1"]),
                    "beeline_aupr": float(beeline_score["aupr"]) if beeline_score else 0.0,
                }
            )
            continue

        hpn_score = score_eval_lookup.get((method, "hpn_dream"))
        beeline_score = score_eval_lookup.get((method, "beeline_gsd"))
        if hpn_score or beeline_score:
            edges_count = int(float(hpn_score["pred_edges"])) if hpn_score else int(float(beeline_score["pred_edges"]))
            rows.append(
                {
                    "method": method,
                    "edges": edges_count,
                    "hpn_precision": 0.0,
                    "hpn_recall": 0.0,
                    "hpn_f1": 0.0,
                    "hpn_aupr": float(hpn_score["aupr"]) if hpn_score else 0.0,
                    "beeline_precision": 0.0,
                    "beeline_recall": 0.0,
                    "beeline_f1": 0.0,
                    "beeline_aupr": float(beeline_score["aupr"]) if beeline_score else 0.0,
                }
            )
            continue

        raise ValueError(f"Unknown method {method}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
