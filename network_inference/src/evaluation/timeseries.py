from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from network_inference.src.evaluation.custom_edges import evaluate_edge_list
from network_inference.src.priors.omnipath import load_omnipath
from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def evaluate_timeseries(config: Dict[str, object]) -> Dict[str, float]:
    paths = config["paths"]
    ts_cfg = dict(config.get("timeseries", {}))

    edges_path = _resolve_optional_path(ts_cfg.get("edges_tsv"), config)
    if not edges_path:
        raise ValueError("timeseries.edges_tsv is required for time-series evaluation")

    mapping_cfg = dict(ts_cfg.get("mapping", {})) if ts_cfg.get("mapping") else None
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
    remove_self = bool(ts_cfg.get("remove_self", True))
    metrics = evaluate_edge_list(
        pred_edges_path=paths["network_edges"],
        truth_edges_path=edges_path,
        processed_h5ad=paths["processed_h5ad"],
        hgnc_alias_tsv=paths.get("hgnc_alias_tsv"),
        truth_source_col=ts_cfg.get("source_col", "source"),
        truth_target_col=ts_cfg.get("target_col", "target"),
        mapping_cfg=mapping_cfg,
        remove_self=remove_self,
    )

    output_path = _resolve_optional_path(ts_cfg.get("output_path"), config)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def build_timeseries_proxy(config: Dict[str, object]) -> Path:
    paths = config["paths"]
    ts_cfg = dict(config.get("timeseries", {}))
    output_path = _resolve_optional_path(ts_cfg.get("edges_tsv"), config)
    if not output_path:
        raise ValueError("timeseries.edges_tsv is required to build a proxy edge list")

    source = str(ts_cfg.get("proxy_source", "omnipath")).lower()
    ensure_mechinterp_path()
    from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))

    if source == "omnipath":
        omnipath_path = paths.get("omnipath_tsv")
        if not omnipath_path:
            raise ValueError("paths.omnipath_tsv is required for OmniPath proxy edges")
        directed_only = bool(config.get("omnipath", {}).get("directed_only", True))
        exclude_composites = bool(
            config.get("omnipath", {}).get("exclude_underscore_composites", False)
        )
        edges = load_omnipath(
            omnipath_path,
            alias_map=alias_map,
            directed_only=directed_only,
            exclude_underscore_composites=exclude_composites,
        )
    elif source == "dorothea":
        ensure_mechinterp_path()
        from src.eval.dorothea import load_dorothea

        dorothea_path = paths.get("dorothea_tsv")
        if not dorothea_path:
            raise ValueError("paths.dorothea_tsv is required for DoRothEA proxy edges")
        edges = load_dorothea(
            dorothea_path,
            confidence_levels=config.get("evaluation", {}).get("dorothea_confidence"),
        )
        edges = normalize_edges(edges, alias_map)
    else:
        raise ValueError(f"Unknown timeseries.proxy_source: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges[["source", "target"]].drop_duplicates().to_csv(output_path, sep="\t", index=False)
    return output_path


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
