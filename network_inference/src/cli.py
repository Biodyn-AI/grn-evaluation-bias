from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from network_inference.src.artifacts.manifest import write_manifest
from network_inference.src.calibration.calibrate import run_calibration
from network_inference.src.evaluation.dorothea_eval import evaluate_dorothea
from network_inference.src.evaluation.perturbation import evaluate_perturbations
from network_inference.src.evaluation.score_edges import evaluate_score_edges
from network_inference.src.evaluation.sweep import run_sweep
from network_inference.src.evaluation.timeseries import build_timeseries_proxy, evaluate_timeseries
from network_inference.src.evidence.head_layer import write_head_layer_evidence
from network_inference.src.inference.attention_network import infer_attention_network
from network_inference.src.priors.omnipath import download_omnipath, download_omnipath_intercell
from network_inference.src.utils.config import load_config, resolve_paths
from network_inference.src.data.loaders import load_processed_anndata


def cmd_infer(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    edges = infer_attention_network(cfg)

    manifest_path = cfg["paths"].get("run_manifest")
    if manifest_path:
        write_manifest(
            manifest_path,
            cfg,
            edges_count=len(edges),
            extra={"command": "infer"},
        )

    output_path = cfg["paths"]["network_edges"]
    print(f"Wrote {len(edges)} edges to {output_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    paths = cfg["paths"]
    metrics = evaluate_dorothea(
        edges_path=paths["network_edges"],
        processed_h5ad=paths["processed_h5ad"],
        dorothea_tsv=paths["dorothea_tsv"],
        hgnc_alias_tsv=paths.get("hgnc_alias_tsv"),
        confidence_levels=cfg.get("evaluation", {}).get("dorothea_confidence"),
    )
    print(json.dumps(metrics, indent=2))


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    result = run_sweep(cfg)
    print(json.dumps(result, indent=2))


def cmd_calibrate(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    report = run_calibration(cfg)
    print(json.dumps(report, indent=2))


def cmd_evaluate_perturbation(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    paths = cfg["paths"]
    pert_cfg = cfg.get("perturbation", {})
    mapping_cfg = dict(pert_cfg.get("mapping", {})) if pert_cfg.get("mapping") else None
    if mapping_cfg:
        if mapping_cfg.get("symbol_map_tsv"):
            mapping_cfg["symbol_map_tsv"] = _resolve_config_relative(
                mapping_cfg.get("symbol_map_tsv"),
                cfg,
            )
        symbol_map_tsvs = mapping_cfg.get("symbol_map_tsvs")
        if symbol_map_tsvs:
            mapping_cfg["symbol_map_tsvs"] = [
                _resolve_config_relative(item, cfg)
                for item in symbol_map_tsvs
                if item
            ]
    metrics = evaluate_perturbations(
        pred_edges_path=paths["network_edges"],
        perturbation_tsv=_resolve_config_relative(pert_cfg.get("edges_tsv"), cfg),
        processed_h5ad=paths["processed_h5ad"],
        hgnc_alias_tsv=paths.get("hgnc_alias_tsv"),
        source_col=pert_cfg.get("source_col", "perturbed_gene"),
        target_col=pert_cfg.get("target_col", "affected_gene"),
        perturbation_col=pert_cfg.get("perturbation_col", "perturbation"),
        mapping_cfg=mapping_cfg,
    )
    output_path = _resolve_config_relative(pert_cfg.get("output_path"), cfg)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def cmd_evaluate_timeseries(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    metrics = evaluate_timeseries(cfg)
    print(json.dumps(metrics, indent=2))


def cmd_evaluate_scores(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    results = evaluate_score_edges(cfg)
    print(json.dumps(results, indent=2))


def cmd_build_timeseries_proxy(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    output_path = build_timeseries_proxy(cfg)
    print(f"Wrote proxy time-series edges to {output_path}")


def cmd_head_layer_evidence(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    paths = cfg["paths"]
    evidence_cfg = cfg.get("evidence", {})

    head_scores = evidence_cfg.get("attention_scores_head_layer") or paths.get("attention_scores_head_layer")
    head_counts = evidence_cfg.get("attention_counts_head_layer") or paths.get("attention_counts_head_layer")
    head_scores = _resolve_config_relative(head_scores, cfg)
    head_counts = _resolve_config_relative(head_counts, cfg)
    output_path = _resolve_config_relative(evidence_cfg.get("output_path"), cfg)
    if not head_scores or not head_counts or not output_path:
        raise ValueError("evidence requires attention_scores_head_layer, attention_counts_head_layer, output_path")

    adata = load_processed_anndata(paths["processed_h5ad"])
    top_k = int(evidence_cfg.get("top_k", 5))
    min_score = evidence_cfg.get("min_score")
    top_n = evidence_cfg.get("top_n_edges")
    count = write_head_layer_evidence(
        edges_path=paths["network_edges"],
        output_path=output_path,
        head_layer_scores_path=head_scores,
        head_layer_counts_path=head_counts,
        gene_names=adata.var_names.values,
        top_k=top_k,
        min_score=min_score,
        top_n=top_n,
    )
    print(f"Wrote head/layer evidence for {count} edges to {output_path}")


def cmd_fetch_omnipath(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    path = cfg["paths"].get("omnipath_tsv")
    if not path:
        raise ValueError("paths.omnipath_tsv must be set to fetch OmniPath data")
    out_path = download_omnipath(path, url=args.url, overwrite=args.overwrite)
    print(f"Downloaded OmniPath interactions to {out_path}")


def cmd_fetch_omnipath_intercell(args: argparse.Namespace) -> None:
    cfg = resolve_paths(load_config(args.config), args.config)
    path = cfg["paths"].get("omnipath_intercell_tsv")
    if not path:
        raise ValueError("paths.omnipath_intercell_tsv must be set to fetch OmniPath intercell data")
    out_path = download_omnipath_intercell(path, url=args.url, overwrite=args.overwrite)
    print(f"Downloaded OmniPath intercell data to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network inference pipeline")
    parser.add_argument("--config", default="configs/base.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Infer network from attention scores")
    infer.set_defaults(func=cmd_infer)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate inferred network against DoRothEA")
    evaluate.set_defaults(func=cmd_evaluate)

    sweep = subparsers.add_parser("sweep", help="Sweep thresholds and report PR metrics")
    sweep.set_defaults(func=cmd_sweep)

    calibrate = subparsers.add_parser("calibrate", help="Calibrate scores to confidence values")
    calibrate.set_defaults(func=cmd_calibrate)

    fetch = subparsers.add_parser("fetch-omnipath", help="Download OmniPath interactions TSV")
    fetch.add_argument("--url", default="https://omnipathdb.org/interactions?format=tsv&genesymbols=1")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.set_defaults(func=cmd_fetch_omnipath)

    fetch_intercell = subparsers.add_parser(
        "fetch-omnipath-intercell", help="Download OmniPath intercell annotations TSV"
    )
    fetch_intercell.add_argument("--url", default="https://omnipathdb.org/intercell?format=tsv")
    fetch_intercell.add_argument("--overwrite", action="store_true")
    fetch_intercell.set_defaults(func=cmd_fetch_omnipath_intercell)

    eval_pert = subparsers.add_parser(
        "evaluate-perturbation", help="Evaluate inferred network against perturbation effects"
    )
    eval_pert.set_defaults(func=cmd_evaluate_perturbation)

    eval_ts = subparsers.add_parser(
        "evaluate-timeseries", help="Evaluate inferred network against time-series edges"
    )
    eval_ts.set_defaults(func=cmd_evaluate_timeseries)

    eval_scores = subparsers.add_parser(
        "evaluate-scores", help="Evaluate scored edge lists against reference networks"
    )
    eval_scores.set_defaults(func=cmd_evaluate_scores)

    build_ts = subparsers.add_parser(
        "build-timeseries-proxy", help="Build a proxy time-series edge list from priors"
    )
    build_ts.set_defaults(func=cmd_build_timeseries_proxy)

    evidence = subparsers.add_parser(
        "evidence-head-layer", help="Annotate edges with head/layer evidence"
    )
    evidence.set_defaults(func=cmd_head_layer_evidence)

    return parser


def _split_config_arg(argv: List[str]) -> Tuple[List[str], str | None]:
    if "--config" not in argv:
        return argv, None
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        raise SystemExit("--config requires a value")
    config_value = argv[idx + 1]
    cleaned = argv[:idx] + argv[idx + 2 :]
    return cleaned, config_value


def _resolve_config_relative(path_value: str | Path | None, config: dict) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    base_dir = config.get("_config_dir")
    if base_dir:
        return (Path(base_dir) / path).resolve()
    return path


def main() -> None:
    parser = build_parser()
    argv, config_override = _split_config_arg(sys.argv[1:])
    args = parser.parse_args(argv)
    if config_override is not None:
        args.config = config_override
    args.func(args)


if __name__ == "__main__":
    main()
