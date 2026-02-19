from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from src.data.preprocess import PreprocessConfig, map_ensembl_to_symbol, preprocess_anndata, subset_to_vocab
from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
from src.data.tabula_sapiens import load_tabula_sapiens
from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import precision_recall_f1
from src.interpret.attention import extract_attention_scores, finalize_attention_scores
from src.model.scgpt_loader import load_scgpt_model
from src.model.vocab import load_vocab
from src.model.wrapper import ScGPTWrapper
from src.network.export import export_edges_tsv
from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config


def _device(device: str | None):
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_scgpt_args(args_path: Path) -> dict:
    with args_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_scgpt_model_args(scgpt_args: dict, vocab) -> dict:
    vocab_map = vocab.gene_to_id
    pad_token = scgpt_args.get("pad_token") or vocab.pad_token
    if not pad_token or pad_token not in vocab_map:
        for token in ("<pad>", "[PAD]", "<PAD>", "PAD"):
            if token in vocab_map:
                pad_token = token
                break
    if not pad_token or pad_token not in vocab_map:
        raise ValueError("Pad token not found in vocab; cannot build model args.")

    return {
        "ntoken": len(vocab_map),
        "d_model": scgpt_args["embsize"],
        "nhead": scgpt_args["nheads"],
        "d_hid": scgpt_args["d_hid"],
        "nlayers": scgpt_args["nlayers"],
        "nlayers_cls": scgpt_args.get("n_layers_cls", 3),
        "n_cls": 1,
        "vocab": vocab_map,
        "dropout": scgpt_args.get("dropout", 0.5),
        "pad_token": pad_token,
        "pad_value": scgpt_args.get("pad_value", 0),
        "do_mvc": bool(scgpt_args.get("MVC", False)),
        "do_dab": False,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
        "input_emb_style": scgpt_args.get("input_emb_style", "continuous"),
        "n_input_bins": scgpt_args.get("n_bins"),
        "cell_emb_style": "avg-pool" if scgpt_args.get("no_cls") else "cls",
        "explicit_zero_prob": False,
        "use_fast_transformer": bool(scgpt_args.get("fast_transformer", False)),
        "fast_transformer_backend": "flash",
        "pre_norm": False,
    }


def _candidate_masks_from_dorothea(
    adata: sc.AnnData,
    paths: dict,
    confidence_levels: list[str] | None,
    use_sources: bool,
    use_targets: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not (use_sources or use_targets):
        return None, None

    dorothea_path = paths.get("dorothea_tsv")
    if not dorothea_path:
        raise ValueError("dorothea_tsv path is required for candidate filtering")

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    true_edges = load_dorothea(dorothea_path, confidence_levels=confidence_levels)
    true_edges = normalize_edges(true_edges, alias_map)

    source_mask = None
    target_mask = None
    if use_sources:
        sources = set(true_edges["source"].unique())
        source_mask = np.array([name in sources for name in gene_names_norm], dtype=bool)
    if use_targets:
        targets = set(true_edges["target"].unique())
        target_mask = np.array([name in targets for name in gene_names_norm], dtype=bool)

    return source_mask, target_mask


def cmd_prepare_data(args):
    cfg = load_config(args.config)
    paths = cfg["paths"]

    adata = load_tabula_sapiens(paths["tabula_sapiens_h5ad"])
    preprocess_cfg = dict(cfg["preprocess"])
    retain_symbols = preprocess_cfg.pop("retain_gene_symbols", None)
    retain_symbols_path = preprocess_cfg.pop("retain_gene_symbols_path", None)
    # Allow large retain lists to be stored in text files instead of inline YAML arrays.
    if retain_symbols_path:
        retain_path = Path(retain_symbols_path)
        if not retain_path.is_absolute():
            retain_path = Path(args.config).resolve().parent / retain_path
        if not retain_path.exists():
            raise FileNotFoundError(f"retain_gene_symbols_path not found: {retain_path}")
        file_symbols = []
        for line in retain_path.read_text(encoding="utf-8").splitlines():
            symbol = line.strip()
            if symbol and not symbol.startswith("#"):
                file_symbols.append(symbol)
        if retain_symbols:
            retain_symbols = list(retain_symbols) + file_symbols
        else:
            retain_symbols = file_symbols
    pre_cfg = PreprocessConfig(**preprocess_cfg)
    if retain_symbols:
        retain_set = {str(symbol).upper() for symbol in retain_symbols if symbol}
        gene_info_path = paths.get("gene_info_csv")
        if gene_info_path:
            gene_info_path = Path(gene_info_path)
        if not gene_info_path or not gene_info_path.exists():
            print("Warning: retain_gene_symbols set but gene_info_csv is missing")
        else:
            mapping_df = pd.read_csv(gene_info_path, usecols=["feature_id", "feature_name"], dtype=str)
            symbol_to_ensembl = {}
            for _, row in mapping_df.iterrows():
                symbol = str(row["feature_name"]).upper()
                ensembl = str(row["feature_id"])
                if symbol and ensembl:
                    symbol_to_ensembl.setdefault(symbol, set()).add(ensembl)

            ensembl_ids = set(pre_cfg.retain_genes or [])
            missing_symbols = []
            for symbol in sorted(retain_set):
                ids = symbol_to_ensembl.get(symbol)
                if not ids:
                    missing_symbols.append(symbol)
                    continue
                ensembl_ids.update(ids)

            pre_cfg.retain_genes = sorted(ensembl_ids)
            if missing_symbols:
                print(f"Warning: {len(missing_symbols)} retain_gene_symbols missing in gene_info_csv")
    adata = preprocess_anndata(adata, pre_cfg)

    missing_in_mapping = []
    gene_info_path = paths.get("gene_info_csv")
    if gene_info_path:
        gene_info_path = Path(gene_info_path)
        if gene_info_path.exists():
            adata, missing_in_mapping = map_ensembl_to_symbol(adata, gene_info_path)
        else:
            print(f"Warning: gene_info_csv not found at {gene_info_path}")

    vocab = load_vocab(paths["scgpt_vocab"])
    adata, missing_in_vocab = subset_to_vocab(adata, vocab.gene_to_id.keys())

    output_path = Path(paths["processed_h5ad"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    if missing_in_mapping:
        mapping_path = output_path.parent / "missing_mapping_genes.txt"
        mapping_path.write_text("\n".join(missing_in_mapping), encoding="utf-8")
    if missing_in_vocab:
        missing_path = output_path.parent / "missing_vocab_genes.txt"
        missing_path.write_text("\n".join(missing_in_vocab), encoding="utf-8")


def cmd_extract_attention(args):
    cfg = load_config(args.config)
    paths = cfg["paths"]
    device = _device(args.device)

    adata = sc.read_h5ad(paths["processed_h5ad"])
    attention_cfg = cfg.get("attention", {})
    reduce_layers = attention_cfg.get("reduce_layers", True)
    reduce_heads = attention_cfg.get("reduce_heads", True)
    max_cells = attention_cfg.get("max_cells")
    sample_strategy = attention_cfg.get("sample_strategy", "head")
    sample_seed = attention_cfg.get("sample_seed")
    if max_cells:
        max_cells = int(max_cells)
        if adata.n_obs > max_cells:
            if sample_strategy == "random":
                if sample_seed is None:
                    sample_seed = cfg.get("project", {}).get("seed", 0)
                rng = np.random.default_rng(int(sample_seed))
                indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
                adata = adata[np.sort(indices)].copy()
            elif sample_strategy == "head":
                adata = adata[:max_cells].copy()
            else:
                raise ValueError(f"Unknown attention.sample_strategy: {sample_strategy}")

    vocab = load_vocab(paths["scgpt_vocab"])
    dataset_cfg = ScGPTDatasetConfig(**cfg["scgpt_dataset"])
    if dataset_cfg.pad_token_id is None:
        if vocab.pad_id is None:
            raise ValueError("pad_token_id is missing and no pad token found in vocab")
        dataset_cfg.pad_token_id = vocab.pad_id
    dataset = ScGPTDataset(adata, vocab.gene_to_id, dataset_cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_scgpt,
    )

    model_cfg = cfg.get("model", {})
    model_args = model_cfg.get("model_args") or {}
    if not model_args:
        args_path = model_cfg.get("args_path")
        if args_path:
            args_path = Path(args_path)
        else:
            args_path = Path(paths["scgpt_checkpoint"]).with_name("args.json")
        if not args_path.exists():
            raise FileNotFoundError(f"scGPT args file not found: {args_path}")
        scgpt_args = _load_scgpt_args(args_path)
        model_args = _build_scgpt_model_args(scgpt_args, vocab)

    model, missing, unexpected = load_scgpt_model(
        entrypoint=cfg["model"]["entrypoint"],
        repo_path=paths["scgpt_repo"],
        checkpoint_path=paths["scgpt_checkpoint"],
        device=device,
        model_args=model_args,
        prefix_to_strip=model_cfg.get("prefix_to_strip"),
    )
    if missing or unexpected:
        print(f"Model load missing keys: {len(missing)} unexpected: {len(unexpected)}")

    model.to(device)
    wrapper = ScGPTWrapper(model, cfg["model"]["forward_key_map"])
    use_head_layer = not (reduce_layers and reduce_heads)
    score_sum_path = None
    score_count_path = None
    if use_head_layer:
        score_sum_path = paths.get("attention_scores_head_layer") or "outputs/atlas/attention_scores_head_layer.npy"
        score_count_path = paths.get("attention_counts_head_layer") or "outputs/atlas/attention_counts.npy"
    score_sum, score_count = extract_attention_scores(
        wrapper,
        dataloader,
        n_genes=adata.n_vars,
        device=device,
        reduce_layers=reduce_layers,
        reduce_heads=reduce_heads,
        score_sum_path=score_sum_path,
        score_count_path=score_count_path,
        score_dtype=attention_cfg.get("score_dtype", np.float32),
        count_dtype=attention_cfg.get("count_dtype", np.int32),
        share_counts=attention_cfg.get("share_counts"),
    )

    if use_head_layer:
        if isinstance(score_sum, np.memmap):
            score_sum.flush()
        if isinstance(score_count, np.memmap):
            score_count.flush()
    else:
        scores_path = Path(paths["attention_scores"])
        counts_path = Path(paths["attention_counts"])
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(scores_path, score_sum)
        np.save(counts_path, score_count)


def cmd_infer_network(args):
    cfg = load_config(args.config)
    paths = cfg["paths"]

    score_sum = np.load(paths["attention_scores"], mmap_mode="r")
    score_count = np.load(paths["attention_counts"], mmap_mode="r")
    if score_sum.ndim != 2:
        raise ValueError(
            "Attention scores must be 2D for network inference. "
            "Run extraction with reduce_layers=true and reduce_heads=true."
        )
    scores = finalize_attention_scores(score_sum, score_count)

    adata = sc.read_h5ad(paths["processed_h5ad"])
    network_cfg_data = dict(cfg.get("network", {}))
    use_sources = bool(network_cfg_data.pop("candidate_sources_from_dorothea", False))
    use_targets = bool(network_cfg_data.pop("candidate_targets_from_dorothea", False))
    network_cfg = NetworkConfig(**network_cfg_data)
    source_mask, target_mask = _candidate_masks_from_dorothea(
        adata,
        paths,
        cfg.get("evaluation", {}).get("dorothea_confidence"),
        use_sources,
        use_targets,
    )
    edges = infer_edges(scores, adata.var_names, network_cfg, source_mask, target_mask)

    export_edges_tsv(edges, paths["network_edges"])


def cmd_evaluate(args):
    cfg = load_config(args.config)
    paths = cfg["paths"]

    pred_edges_df = pd.read_csv(paths["network_edges"], sep="\t")
    pred_edges_df = pred_edges_df[["source", "target"]]
    pred_edges_df = pred_edges_df.drop_duplicates()

    true_edges = load_dorothea(
        paths["dorothea_tsv"],
        confidence_levels=cfg.get("evaluation", {}).get("dorothea_confidence"),
    )
    adata = sc.read_h5ad(paths["processed_h5ad"])
    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    gene_set = set(gene_names_norm)
    pred_edges_df = normalize_edges(pred_edges_df, alias_map)
    pred_edges_df = pred_edges_df[
        pred_edges_df["source"].isin(gene_set) & pred_edges_df["target"].isin(gene_set)
    ].drop_duplicates()
    true_edges = normalize_edges(true_edges, alias_map)
    true_edges = true_edges[
        true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
    ].drop_duplicates()

    metrics = precision_recall_f1(pred_edges_df, true_edges)
    print(metrics)


def build_parser():
    parser = argparse.ArgumentParser(description="scGPT mechanistic interpretability pipeline")
    parser.add_argument("--config", default="configs/base.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Preprocess Tabula Sapiens data")
    prepare.set_defaults(func=cmd_prepare_data)

    extract = subparsers.add_parser("extract-attention", help="Extract attention scores")
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--device", default=None)
    extract.set_defaults(func=cmd_extract_attention)

    infer = subparsers.add_parser("infer-network", help="Infer gene network")
    infer.set_defaults(func=cmd_infer_network)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate inferred network against ground truth")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
