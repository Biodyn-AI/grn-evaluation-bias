from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from network_inference.src.data.mapping import (
    collect_symbol_map_paths,
    collect_external_ids,
    extend_alias_map_from_adata,
    extend_alias_map_from_mygene,
    extend_alias_map_from_symbol_map,
    load_hgnc_alias_map_extended,
    map_edge_table,
)
from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def evaluate_score_edges(config: Dict[str, object]) -> List[Dict[str, object]]:
    paths = config["paths"]
    eval_cfg = dict(config.get("score_eval", {}))
    remove_self = bool(eval_cfg.get("remove_self", True))
    overlap_gate_cfg = eval_cfg.get("overlap_gate") or None

    gene_names_norm, alias_map = _load_gene_names(
        paths,
        eval_cfg.get("mapping"),
        read_h5ad_backed=bool(eval_cfg.get("read_h5ad_backed", True)),
    )
    gene_set = set(gene_names_norm)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_norm)}
    max_genes_for_matrix = eval_cfg.get("max_genes_for_matrix")
    use_matrix_metrics = True
    if max_genes_for_matrix is not None and len(gene_names_norm) > int(max_genes_for_matrix):
        use_matrix_metrics = False
    candidate_mask = None
    if use_matrix_metrics:
        candidate_mask = np.ones((len(gene_names_norm), len(gene_names_norm)), dtype=bool)
        if remove_self:
            candidate_mask &= ~np.eye(len(gene_names_norm), dtype=bool)

    reference_defs = eval_cfg.get("references", [])
    if not reference_defs:
        raise ValueError("score_eval.references is required")

    methods = eval_cfg.get("methods", [])
    if not methods:
        raise ValueError("score_eval.methods is required")

    raw_references = []
    for ref in reference_defs:
        ref_path = _resolve_optional_path(ref.get("edges_tsv"), config)
        if not ref_path:
            raise ValueError("score_eval.references.edges_tsv is required")
        mapping_cfg = dict(ref.get("mapping", {})) if ref.get("mapping") else None
        if mapping_cfg:
            if mapping_cfg.get("symbol_map_tsv"):
                mapping_cfg["symbol_map_tsv"] = _resolve_optional_path(
                    mapping_cfg.get("symbol_map_tsv"),
                    config,
                )
            symbol_map_tsvs = mapping_cfg.get("symbol_map_tsvs")
            if symbol_map_tsvs:
                mapping_cfg["symbol_map_tsvs"] = [
                    _resolve_optional_path(item, config)
                    for item in symbol_map_tsvs
                    if item
                ]
        ref_edges = _load_edge_table(
            ref_path,
            ref.get("source_col", "source"),
            ref.get("target_col", "target"),
            score_col=None,
        )
        raw_references.append(
            {
                "name": ref.get("name", "reference"),
                "edges": ref_edges,
                "mapping_cfg": mapping_cfg,
            }
        )

    raw_methods = []
    for method in methods:
        method_name = method.get("name", "method")
        edges_path = _resolve_optional_path(method.get("edges_tsv"), config)
        if not edges_path:
            raise ValueError("score_eval.methods.edges_tsv is required")
        pred_edges = _load_edge_table(
            edges_path,
            method.get("source_col", "source"),
            method.get("target_col", "target"),
            score_col=method.get("score_col", "score"),
        )
        mapping_cfg = dict(method.get("mapping", {})) if method.get("mapping") else None
        if mapping_cfg:
            if mapping_cfg.get("symbol_map_tsv"):
                mapping_cfg["symbol_map_tsv"] = _resolve_optional_path(
                    mapping_cfg.get("symbol_map_tsv"),
                    config,
                )
            symbol_map_tsvs = mapping_cfg.get("symbol_map_tsvs")
            if symbol_map_tsvs:
                mapping_cfg["symbol_map_tsvs"] = [
                    _resolve_optional_path(item, config)
                    for item in symbol_map_tsvs
                    if item
                ]
        raw_methods.append(
            {
                "name": method_name,
                "edges": pred_edges,
                "mapping_cfg": mapping_cfg,
                "score_col": method.get("score_col", "score"),
            }
        )

    id_mapping_cfg = (eval_cfg.get("mapping") or {}).get("id_mapping")
    symbol_map_paths = set()
    for item in raw_references + raw_methods:
        mapping_cfg = item.get("mapping_cfg")
        for symbol_map_path in collect_symbol_map_paths(mapping_cfg):
            symbol_map_paths.add(symbol_map_path)
    for symbol_map_path in sorted(symbol_map_paths):
        alias_map = extend_alias_map_from_symbol_map(
            alias_map,
            symbol_map_path,
            id_mapping_cfg,
        )
    if id_mapping_cfg:
        id_values: List[str] = []
        for item in raw_references + raw_methods:
            id_values.extend(collect_external_ids(item["edges"]["source"]))
            id_values.extend(collect_external_ids(item["edges"]["target"]))
        alias_map = extend_alias_map_from_mygene(alias_map, id_values, id_mapping_cfg)

    diagnostics_cfg = eval_cfg.get("diagnostics")
    if diagnostics_cfg:
        _write_mapping_diagnostics(
            raw_references,
            alias_map,
            gene_set,
            diagnostics_cfg,
            config,
        )

    references = []
    for ref in raw_references:
        ref_edges = _normalize_edges(ref["edges"], alias_map, ref["mapping_cfg"])
        overlap_stats = _compute_overlap_stats(ref_edges, gene_set)
        excluded = False
        if overlap_gate_cfg:
            min_ref_pct = float(overlap_gate_cfg.get("min_ref_node_overlap_pct", 0.0))
            min_gene_pct = float(overlap_gate_cfg.get("min_gene_universe_overlap_pct", 0.0))
            excluded = (
                overlap_stats["ref_node_overlap_pct"] < min_ref_pct
                or overlap_stats["gene_universe_overlap_pct"] < min_gene_pct
            )

        filtered_edges = ref_edges
        true_mask = None
        if not excluded:
            filtered_edges = ref_edges[
                ref_edges["source"].isin(gene_set) & ref_edges["target"].isin(gene_set)
            ].drop_duplicates()
            if use_matrix_metrics:
                true_mask = np.zeros((len(gene_names_norm), len(gene_names_norm)), dtype=bool)
                for _, row in filtered_edges.iterrows():
                    src = gene_to_idx.get(row["source"])
                    tgt = gene_to_idx.get(row["target"])
                    if src is None or tgt is None:
                        continue
                    true_mask[src, tgt] = True
        references.append(
            {
                "name": ref["name"],
                "edges": filtered_edges,
                "true_mask": true_mask,
                "excluded": excluded,
                "overlap_stats": overlap_stats,
            }
        )

    results: List[Dict[str, object]] = []
    ensure_mechinterp_path()
    from network_inference.src.evaluation.metrics import aupr
    from src.eval.metrics import precision_recall_f1

    candidate_edges = None
    if use_matrix_metrics:
        candidate_edges = int(candidate_mask.sum())
    else:
        n_genes = len(gene_names_norm)
        candidate_edges = int(n_genes * (n_genes - 1)) if remove_self else int(n_genes * n_genes)

    for method in raw_methods:
        method_name = method["name"]
        pred_edges = _normalize_edges(method["edges"], alias_map, method["mapping_cfg"])
        pred_edges = pred_edges[
            pred_edges["source"].isin(gene_set) & pred_edges["target"].isin(gene_set)
        ].drop_duplicates()

        if pred_edges.empty:
            for ref in references:
                if ref["excluded"]:
                    results.append(
                        {
                            "method": method_name,
                            "reference": ref["name"],
                            "precision": None,
                            "recall": None,
                            "f1": None,
                            "aupr": None,
                            "pred_edges": 0,
                            "true_edges": len(ref["edges"]),
                            "candidate_edges": candidate_edges,
                            "reference_excluded": True,
                            **ref["overlap_stats"],
                        }
                    )
                    continue
                results.append(
                    {
                        "method": method_name,
                        "reference": ref["name"],
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "aupr": None if not use_matrix_metrics else 0.0,
                        "pred_edges": 0,
                        "true_edges": len(ref["edges"]),
                        "candidate_edges": candidate_edges,
                        "reference_excluded": False,
                        **ref["overlap_stats"],
                    }
                )
            continue

        scores = None
        if use_matrix_metrics:
            scores = np.zeros((len(gene_names_norm), len(gene_names_norm)), dtype=np.float32)
            for _, row in pred_edges.iterrows():
                src = gene_to_idx.get(row["source"])
                tgt = gene_to_idx.get(row["target"])
                if src is None or tgt is None:
                    continue
                score = float(row["score"]) if "score" in row else 1.0
                if score > scores[src, tgt]:
                    scores[src, tgt] = score

        for ref in references:
            if ref["excluded"]:
                results.append(
                    {
                        "method": method_name,
                        "reference": ref["name"],
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "aupr": None,
                        "pred_edges": int(pred_edges.shape[0]),
                        "true_edges": len(ref["edges"]),
                        "candidate_edges": candidate_edges,
                        "reference_excluded": True,
                        **ref["overlap_stats"],
                    }
                )
                continue
            prf = precision_recall_f1(
                pred_edges[["source", "target"]],
                ref["edges"][["source", "target"]],
            )
            aupr_value = None
            if use_matrix_metrics:
                scores_flat = scores[candidate_mask]
                labels_flat = ref["true_mask"][candidate_mask].astype(np.int8)
                aupr_value = float(aupr(scores_flat, labels_flat))
            results.append(
                {
                    "method": method_name,
                    "reference": ref["name"],
                    "precision": prf["precision"],
                    "recall": prf["recall"],
                    "f1": prf["f1"],
                    "aupr": aupr_value,
                    "pred_edges": int(pred_edges.shape[0]),
                    "true_edges": int(ref["true_mask"].sum())
                    if ref["true_mask"] is not None
                    else int(ref["edges"].shape[0]),
                    "candidate_edges": candidate_edges,
                    "reference_excluded": False,
                    **ref["overlap_stats"],
                }
            )

    output_path = _resolve_optional_path(eval_cfg.get("output_path"), config)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        csv_path = output_path.with_suffix(".csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)

    return results


def _load_gene_names(
    paths: Dict[str, Path],
    mapping_cfg: Dict[str, object] | None,
    read_h5ad_backed: bool = True,
) -> Tuple[np.ndarray, Dict[str, str]]:
    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_gene_names

    backed = "r" if read_h5ad_backed else None
    adata = sc.read_h5ad(Path(paths["processed_h5ad"]), backed=backed)
    extra_cols = mapping_cfg.get("hgnc_extra_alias_cols") if mapping_cfg else None
    alias_map = load_hgnc_alias_map_extended(paths.get("hgnc_alias_tsv"), extra_cols)
    if mapping_cfg:
        alias_map = extend_alias_map_from_adata(
            alias_map,
            adata,
            mapping_cfg.get("adata_alias_cols"),
        )
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return gene_names_norm, alias_map


def _load_edge_table(
    path: Path,
    source_col: str,
    target_col: str,
    score_col: str | None,
) -> pd.DataFrame:
    def coerce_col(value: str | int | None) -> str | int | None:
        if isinstance(value, int) or value is None:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return value

    source_col = coerce_col(source_col)
    target_col = coerce_col(target_col)
    score_col = coerce_col(score_col)
    header = None if any(isinstance(col, int) for col in (source_col, target_col, score_col)) else "infer"
    df = pd.read_csv(path, sep="\t", header=header)
    required = {source_col, target_col}
    if score_col:
        required.add(score_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    cols = [source_col, target_col]
    if score_col:
        cols.append(score_col)
    edges = df[cols].dropna().copy()
    edges.columns = ["source", "target"] + (["score"] if score_col else [])
    return edges


def _normalize_edges(
    edges: pd.DataFrame,
    alias_map: Dict[str, str],
    mapping_cfg: Dict[str, object] | None,
) -> pd.DataFrame:
    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_edges

    edges = edges.copy()
    if mapping_cfg:
        edges = map_edge_table(edges, mapping_cfg, alias_map)
    edges = normalize_edges(edges, alias_map)
    return edges


def _resolve_optional_path(path_value: str | Path | None, config: Dict[str, object]) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    base_dir = config.get("_config_dir")
    if base_dir:
        return (Path(base_dir) / path).resolve()
    return path


def _normalize_edges_for_diagnostics(
    edges: pd.DataFrame,
    alias_map: Dict[str, str],
    mapping_cfg: Dict[str, object] | None,
) -> pd.DataFrame:
    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_edges

    edges = edges.copy()
    if mapping_cfg:
        mapped_cfg = dict(mapping_cfg)
        mapped_cfg["drop_unmapped"] = False
        edges = map_edge_table(edges, mapped_cfg, alias_map)
    return normalize_edges(edges, alias_map)


def _compute_overlap_stats(edges: pd.DataFrame, gene_set: set[str]) -> Dict[str, float]:
    sources = edges["source"] if "source" in edges else pd.Series([], dtype=str)
    targets = edges["target"] if "target" in edges else pd.Series([], dtype=str)
    ref_nodes = set(pd.concat([sources, targets], ignore_index=True).dropna().unique())
    overlap_nodes = ref_nodes.intersection(gene_set)
    ref_node_count = len(ref_nodes)
    overlap_count = len(overlap_nodes)
    ref_overlap_pct = (overlap_count / ref_node_count * 100.0) if ref_node_count else 0.0
    gene_overlap_pct = (overlap_count / len(gene_set) * 100.0) if gene_set else 0.0
    return {
        "ref_nodes": ref_node_count,
        "overlap_nodes": overlap_count,
        "ref_node_overlap_pct": ref_overlap_pct,
        "gene_universe_overlap_pct": gene_overlap_pct,
    }


def _write_mapping_diagnostics(
    raw_references: List[Dict[str, object]],
    alias_map: Dict[str, str],
    gene_set: set[str],
    diagnostics_cfg: Dict[str, object],
    config: Dict[str, object],
) -> None:
    ref_filter = diagnostics_cfg.get("references") or []
    if isinstance(ref_filter, str):
        ref_filter = [ref_filter]
    ref_filter_set = {name for name in ref_filter if name}
    top_n = int(diagnostics_cfg.get("top_n", 50))
    report_path = _resolve_optional_path(diagnostics_cfg.get("output_path"), config)
    tsv_path = _resolve_optional_path(diagnostics_cfg.get("tsv_path"), config)

    report_rows: List[Dict[str, object]] = []
    tsv_rows: List[Dict[str, object]] = []

    for ref in raw_references:
        ref_name = ref.get("name", "reference")
        if ref_filter_set and ref_name not in ref_filter_set:
            continue
        normalized = _normalize_edges_for_diagnostics(
            ref["edges"],
            alias_map,
            ref.get("mapping_cfg"),
        )
        sources = normalized["source"]
        targets = normalized["target"]
        missing_sources = sources[~sources.isin(gene_set)]
        missing_targets = targets[~targets.isin(gene_set)]
        missing_all = pd.concat([missing_sources, missing_targets], ignore_index=True)
        missing_counts = missing_all.value_counts()

        report_rows.append(
            {
                "reference": ref_name,
                "total_edges_raw": int(ref["edges"].shape[0]),
                "total_edges_normalized": int(normalized.shape[0]),
                "unique_sources_normalized": int(sources.nunique()),
                "unique_targets_normalized": int(targets.nunique()),
                "missing_sources": int(missing_sources.shape[0]),
                "missing_targets": int(missing_targets.shape[0]),
                "missing_unique": int(missing_counts.shape[0]),
                "missing_total": int(missing_all.shape[0]),
                "missing_top": [
                    {"id": idx, "count": int(count)}
                    for idx, count in missing_counts.head(top_n).items()
                ],
            }
        )

        for idx, count in missing_counts.head(top_n).items():
            tsv_rows.append(
                {
                    "reference": ref_name,
                    "role": "any",
                    "missing_id": idx,
                    "count": int(count),
                }
            )

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")

    if tsv_path and tsv_rows:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tsv_rows).to_csv(tsv_path, sep="\t", index=False)
