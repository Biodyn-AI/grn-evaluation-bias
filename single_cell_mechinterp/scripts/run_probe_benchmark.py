from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import canonical_symbol, load_hgnc_alias_map, normalize_edges, normalize_gene_names
from src.eval.metrics import aupr, precision_recall_f1
from src.interpret.attention import extract_attention_scores, finalize_attention_scores
from src.interpret.attribution import integrated_gradients
from src.model.scgpt_loader import load_scgpt_model
from src.model.vocab import load_vocab
from src.model.wrapper import ScGPTWrapper
from src.network.export import export_edges_tsv
from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config
from src.utils.torch_utils import move_to_device


def _device(device: str | None) -> str:
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


def _extract_output_tensor(outputs, output_key: str | None) -> torch.Tensor:
    if output_key:
        if isinstance(outputs, dict) and output_key in outputs:
            return outputs[output_key]
        if hasattr(outputs, output_key):
            return getattr(outputs, output_key)
        raise KeyError(f"Output key '{output_key}' not found in model outputs")

    if isinstance(outputs, dict):
        for key in ("mlm_output", "mvc_output", "logits", "pred", "output"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
        for value in outputs.values():
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for value in outputs:
            if torch.is_tensor(value):
                return value
    raise ValueError("Unable to extract a tensor output from the model outputs")


def _reduce_output(output: torch.Tensor, reduce_mode: str | None) -> torch.Tensor:
    if output.dim() == 2:
        return output
    if output.dim() != 3:
        raise ValueError(f"Expected output with 2 or 3 dims, got {output.shape}")

    if reduce_mode is None or reduce_mode == "none":
        raise ValueError("output_reduce must be set for 3D outputs")
    if reduce_mode == "mean":
        return output.mean(dim=-1)
    if reduce_mode == "sum":
        return output.sum(dim=-1)
    raise ValueError(f"Unsupported output_reduce: {reduce_mode}")


def _select_positions(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    top_k: int | None,
    use_abs: bool,
) -> np.ndarray:
    if use_abs:
        scores = scores.abs()
    valid_scores = scores[valid_mask]
    if valid_scores.numel() == 0:
        return np.array([], dtype=np.int64)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if top_k is None or top_k >= valid_scores.numel():
        return valid_indices.detach().cpu().numpy()
    if top_k <= 0:
        return np.array([], dtype=np.int64)
    top = torch.topk(valid_scores, k=top_k).indices
    return valid_indices[top].detach().cpu().numpy()


def _accumulate_vector(
    score_sum: np.ndarray,
    score_count: np.ndarray,
    gene_indices: np.ndarray,
    source_positions: np.ndarray,
    target_gene_idx: int,
    values: np.ndarray,
) -> None:
    if source_positions.size == 0:
        return
    source_gene_idx = gene_indices[source_positions]
    valid_mask = source_gene_idx >= 0
    if not np.any(valid_mask):
        return
    source_gene_idx = source_gene_idx[valid_mask]
    values = values[source_positions][valid_mask]
    np.add.at(score_sum, (source_gene_idx, target_gene_idx), values)
    np.add.at(score_count, (source_gene_idx, target_gene_idx), 1)


def _finalize_scores(score_sum: np.ndarray, score_count: np.ndarray) -> np.ndarray:
    denom = np.maximum(score_count, 1)
    return score_sum / denom


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _normalize_scores(scores: np.ndarray, method: str | None, axis: str) -> np.ndarray:
    if method is None or method == "none":
        return scores

    if axis == "target":
        return _normalize_scores(scores.T, method, "source").T
    if axis == "global":
        flat = scores.ravel()
        if method == "rank":
            ranks = _rankdata(flat)
            if len(ranks) > 1:
                ranks = (ranks - 1) / (len(ranks) - 1)
            return ranks.reshape(scores.shape).astype(np.float32)
        if method == "zscore":
            mean = np.mean(flat)
            std = np.std(flat)
            if std == 0:
                return np.zeros_like(scores)
            return ((scores - mean) / std).astype(np.float32)
        raise ValueError(f"Unsupported normalization: {method}")

    if axis != "source":
        raise ValueError(f"Unsupported normalization_axis: {axis}")

    normalized = np.zeros_like(scores, dtype=np.float32)
    for idx, row in enumerate(scores):
        if method == "rank":
            ranks = _rankdata(row)
            if len(ranks) > 1:
                ranks = (ranks - 1) / (len(ranks) - 1)
            normalized[idx] = ranks
        elif method == "zscore":
            mean = np.mean(row)
            std = np.std(row)
            if std == 0:
                normalized[idx] = 0.0
            else:
                normalized[idx] = (row - mean) / std
        else:
            raise ValueError(f"Unsupported normalization: {method}")
    return normalized


def _load_gene_list(path: str | None, alias_map: Dict[str, str]) -> List[str]:
    if not path:
        return []
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    genes = []
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        gene = line.split()[0]
        gene = canonical_symbol(gene, alias_map)
        if gene:
            genes.append(gene)
    return genes


def _resolve_candidates(
    gene_names_norm: np.ndarray,
    candidate_names: List[str],
) -> Tuple[List[str], List[int]]:
    gene_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(gene_names_norm):
        if name and name not in gene_to_idx:
            gene_to_idx[name] = idx

    if candidate_names:
        seen = set()
        ordered = []
        for name in candidate_names:
            if name in gene_to_idx and name not in seen:
                ordered.append(name)
                seen.add(name)
        names = ordered
    else:
        names = list(gene_to_idx.keys())

    indices = [gene_to_idx[name] for name in names]
    return names, indices


def _build_label_matrix(
    true_edges: pd.DataFrame,
    source_names: List[str],
    target_names: List[str],
) -> np.ndarray:
    source_index = {name: i for i, name in enumerate(source_names)}
    target_index = {name: i for i, name in enumerate(target_names)}
    labels = np.zeros((len(source_names), len(target_names)), dtype=np.int8)
    for _, row in true_edges.iterrows():
        i = source_index.get(row["source"])
        j = target_index.get(row["target"])
        if i is not None and j is not None:
            labels[i, j] = 1
    return labels


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.ravel()
    labels = labels.ravel().astype(bool)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    ranks = _rankdata(scores)
    sum_pos = ranks[labels].sum()
    return float((sum_pos - pos * (pos + 1) / 2) / (pos * neg))


def _precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> Tuple[float, float]:
    scores_flat = scores.ravel()
    labels_flat = labels.ravel().astype(bool)
    if scores_flat.size == 0:
        return 0.0, 0.0
    k = min(k, scores_flat.size)
    if k <= 0:
        return 0.0, 0.0
    top_idx = np.argpartition(-scores_flat, k - 1)[:k]
    tp = labels_flat[top_idx].sum()
    total_pos = labels_flat.sum()
    precision = float(tp / k) if k else 0.0
    recall = float(tp / total_pos) if total_pos else 0.0
    return precision, recall


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return 0.0
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    if np.std(rank_a) == 0 or np.std(rank_b) == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _top_edges_global(
    scores: np.ndarray,
    source_names: List[str],
    target_names: List[str],
    source_indices: List[int],
    target_indices: List[int],
    k: int,
    remove_self: bool,
) -> set[tuple[str, str]]:
    if not source_indices or not target_indices or k <= 0:
        return set()
    sub = scores[np.ix_(source_indices, target_indices)].copy()
    if remove_self:
        target_index = {name: j for j, name in enumerate(target_names)}
        for i, name in enumerate(source_names):
            j = target_index.get(name)
            if j is not None:
                sub[i, j] = -np.inf
    flat = sub.ravel()
    k = min(k, flat.size)
    top_idx = np.argpartition(-flat, k - 1)[:k]
    source_rel, target_rel = np.unravel_index(top_idx, sub.shape)
    edges = set()
    for s_idx, t_idx in zip(source_rel, target_rel):
        if not np.isfinite(sub[s_idx, t_idx]):
            continue
        edges.add((source_names[s_idx], target_names[t_idx]))
    return edges


def _compute_attention_scores(
    wrapper: ScGPTWrapper,
    dataloader: DataLoader,
    n_genes: int,
    device: str,
    attention_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    score_sum, score_count = extract_attention_scores(
        wrapper,
        dataloader,
        n_genes=n_genes,
        device=device,
        reduce_layers=attention_cfg.get("reduce_layers", True),
        reduce_heads=attention_cfg.get("reduce_heads", True),
    )
    return score_sum, score_count


def _compute_attribution_scores(
    probe: str,
    wrapper: ScGPTWrapper,
    dataloader: DataLoader,
    n_genes: int,
    device: str,
    output_key: str | None,
    output_reduce: str | None,
    target_top_k: int | None,
    source_top_k: int | None,
    target_selection: str,
    source_selection: str,
    selection_use_abs: bool,
    baseline_value: float,
    ig_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    score_sum = np.zeros((n_genes, n_genes), dtype=np.float32)
    score_count = np.zeros((n_genes, n_genes), dtype=np.int32)

    wrapper.eval()
    for batch in dataloader:
        batch = move_to_device(batch, device)
        batch_size = batch["gene_ids"].shape[0]
        for idx in range(batch_size):
            sample = {key: value[idx : idx + 1] for key, value in batch.items()}
            gene_indices = sample["gene_indices"][0].detach().cpu().numpy()
            valid_mask = sample["gene_indices"][0] != -1
            if not torch.any(valid_mask):
                continue

            if probe == "perturbation":
                gene_values = sample["gene_values"]
                with torch.no_grad():
                    outputs = wrapper.forward(sample)
                    output = _reduce_output(_extract_output_tensor(outputs, output_key), output_reduce)
                target_scores = output[0] if target_selection == "output" else gene_values[0]
                source_scores = gene_values[0] if source_selection == "expression" else output[0]
                target_positions = _select_positions(
                    target_scores, valid_mask, target_top_k, selection_use_abs
                )
                source_positions = _select_positions(
                    source_scores, valid_mask, source_top_k, selection_use_abs
                )
                if target_positions.size == 0 or source_positions.size == 0:
                    continue
                base_output = output[0].detach().cpu().numpy()

                for source_pos in source_positions:
                    source_gene_idx = gene_indices[source_pos]
                    if source_gene_idx < 0:
                        continue
                    perturbed = gene_values.clone()
                    perturbed[0, source_pos] = baseline_value
                    sample_inputs = dict(sample)
                    sample_inputs["gene_values"] = perturbed
                    with torch.no_grad():
                        pert_out = _reduce_output(
                            _extract_output_tensor(wrapper.forward(sample_inputs), output_key),
                            output_reduce,
                        )[0]
                    delta = base_output - pert_out.detach().cpu().numpy()
                    for target_pos in target_positions:
                        target_gene_idx = gene_indices[target_pos]
                        if target_gene_idx < 0:
                            continue
                        score_sum[source_gene_idx, target_gene_idx] += float(delta[target_pos])
                        score_count[source_gene_idx, target_gene_idx] += 1
                continue

            gene_values = sample["gene_values"].clone().detach().requires_grad_(True)
            sample_inputs = dict(sample)
            sample_inputs["gene_values"] = gene_values
            outputs = wrapper.forward(sample_inputs)
            output = _reduce_output(_extract_output_tensor(outputs, output_key), output_reduce)
            target_scores = output[0] if target_selection == "output" else gene_values[0]
            source_scores = gene_values[0] if source_selection == "expression" else output[0]
            target_positions = _select_positions(
                target_scores, valid_mask, target_top_k, selection_use_abs
            )
            source_positions = _select_positions(
                source_scores, valid_mask, source_top_k, selection_use_abs
            )
            if target_positions.size == 0 or source_positions.size == 0:
                continue

            for t_idx, target_pos in enumerate(target_positions):
                target_gene_idx = gene_indices[target_pos]
                if target_gene_idx < 0:
                    continue
                if probe == "integrated_gradients":
                    baseline = torch.full_like(gene_values, baseline_value)

                    def model_fn(values: torch.Tensor) -> torch.Tensor:
                        inner_inputs = dict(sample_inputs)
                        inner_inputs["gene_values"] = values
                        inner_out = _reduce_output(
                            _extract_output_tensor(wrapper.forward(inner_inputs), output_key),
                            output_reduce,
                        )
                        return inner_out[0, target_pos]

                    attr = integrated_gradients(gene_values, model_fn, baseline=baseline, steps=ig_steps)[
                        0
                    ]
                else:
                    retain = t_idx < len(target_positions) - 1
                    grads = torch.autograd.grad(
                        output[0, target_pos],
                        gene_values,
                        retain_graph=retain,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if grads is None:
                        continue
                    attr = grads[0]
                    if probe == "grad_input":
                        attr = attr * gene_values[0]

                attr_np = attr.detach().cpu().numpy()
                _accumulate_vector(
                    score_sum,
                    score_count,
                    gene_indices,
                    source_positions,
                    target_gene_idx,
                    attr_np,
                )

    return score_sum, score_count


def run_benchmark(config_path: str, device: str | None, probe_filter: List[str] | None) -> None:
    cfg = load_config(config_path)
    project_seed = cfg.get("project", {}).get("seed")
    if project_seed is not None:
        np.random.seed(project_seed)
        torch.manual_seed(project_seed)

    paths = cfg["paths"]
    probe_cfg = cfg.get("probe_benchmark", {})
    probes = probe_cfg.get("probes", [])
    if probe_filter:
        probes = [probe for probe in probes if probe in probe_filter]

    output_dir = Path(paths.get("probe_output_dir", "outputs/probe_benchmark"))
    matrices_dir = Path(paths.get("probe_matrices_dir", output_dir / "matrices"))
    edges_dir = Path(paths.get("probe_edges_dir", output_dir / "edges"))
    metrics_path = Path(paths.get("probe_metrics_csv", output_dir / "probe_metrics.csv"))
    agreement_path = Path(paths.get("probe_agreement_tsv", output_dir / "probe_agreement.tsv"))
    recommendations_path = Path(
        paths.get("probe_recommendations_md", output_dir / "recommendations.md")
    )
    matrices_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(paths["processed_h5ad"])
    max_cells = probe_cfg.get("max_cells")
    sample_mode = probe_cfg.get("sample_cells")
    if sample_mode and not max_cells:
        raise ValueError("sample_cells requires max_cells to be set")
    if max_cells:
        max_cells = int(max_cells)
        if sample_mode:
            if sample_mode not in {"random", "bootstrap"}:
                raise ValueError(f"Unsupported sample_cells: {sample_mode}")
            seed = probe_cfg.get("sample_seed", project_seed)
            rng = np.random.default_rng(seed)
            if sample_mode == "random":
                size = min(max_cells, adata.n_obs)
                indices = rng.choice(adata.n_obs, size=size, replace=False)
            else:
                indices = rng.choice(adata.n_obs, size=max_cells, replace=True)
            adata = adata[indices].copy()
        elif adata.n_obs > max_cells:
            adata = adata[:max_cells].copy()

    vocab = load_vocab(paths["scgpt_vocab"])
    dataset_cfg = ScGPTDatasetConfig(**cfg["scgpt_dataset"])
    if dataset_cfg.pad_token_id is None:
        if vocab.pad_id is None:
            raise ValueError("pad_token_id is missing and no pad token found in vocab")
        dataset_cfg.pad_token_id = vocab.pad_id
    dataset = ScGPTDataset(adata, vocab.gene_to_id, dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=int(probe_cfg.get("batch_size", 1)),
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

    device = _device(device)
    model, missing, unexpected = load_scgpt_model(
        entrypoint=model_cfg["entrypoint"],
        repo_path=paths["scgpt_repo"],
        checkpoint_path=paths["scgpt_checkpoint"],
        device=device,
        model_args=model_args,
        prefix_to_strip=model_cfg.get("prefix_to_strip"),
    )
    if missing or unexpected:
        print(f"Model load missing keys: {len(missing)} unexpected: {len(unexpected)}")
    model.to(device)
    wrapper = ScGPTWrapper(model, model_cfg["forward_key_map"])

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)
    candidate_sources = _load_gene_list(probe_cfg.get("candidate_sources_path"), alias_map)
    candidate_targets = _load_gene_list(probe_cfg.get("candidate_targets_path"), alias_map)
    source_names, source_indices = _resolve_candidates(gene_names_norm, candidate_sources)
    target_names, target_indices = _resolve_candidates(gene_names_norm, candidate_targets)

    references_cfg = probe_cfg.get("references")
    if not references_cfg:
        references_cfg = [
            {
                "name": "reference",
                "path": paths["dorothea_tsv"],
                "confidence_levels": cfg.get("evaluation", {}).get("dorothea_confidence"),
            }
        ]

    network_cfg = NetworkConfig(**cfg.get("network", {}))

    if not source_indices or not target_indices:
        raise ValueError("Candidate sources or targets are empty; check probe_benchmark config")

    metrics_rows = []
    probe_scores: Dict[str, np.ndarray] = {}
    allowed_probes = {"attention", "grad_input", "integrated_gradients", "perturbation"}
    for probe in probes:
        if probe not in allowed_probes:
            raise ValueError(f"Unsupported probe '{probe}', expected one of {sorted(allowed_probes)}")
        print(f"Running probe: {probe}")
        if probe == "attention":
            score_sum, score_count = _compute_attention_scores(
                wrapper, dataloader, adata.n_vars, device, cfg.get("attention", {})
            )
            raw_scores = finalize_attention_scores(score_sum, score_count)
        else:
            score_sum, score_count = _compute_attribution_scores(
                probe=probe,
                wrapper=wrapper,
                dataloader=dataloader,
                n_genes=adata.n_vars,
                device=device,
                output_key=probe_cfg.get("output_key"),
                output_reduce=probe_cfg.get("output_reduce"),
                target_top_k=probe_cfg.get("target_top_k"),
                source_top_k=probe_cfg.get("source_top_k"),
                target_selection=probe_cfg.get("target_selection", "output"),
                source_selection=probe_cfg.get("source_selection", "expression"),
                selection_use_abs=probe_cfg.get("selection_use_abs", False),
                baseline_value=float(probe_cfg.get("baseline_value", 0.0)),
                ig_steps=int(probe_cfg.get("ig_steps", 50)),
            )
            raw_scores = _finalize_scores(score_sum, score_count)

        if probe_cfg.get("use_abs", False):
            raw_scores = np.abs(raw_scores)
        scores = _normalize_scores(
            raw_scores,
            method=probe_cfg.get("normalization", "none"),
            axis=probe_cfg.get("normalization_axis", "source"),
        )

        np.save(matrices_dir / f"{probe}_raw.npy", raw_scores)
        np.save(matrices_dir / f"{probe}_counts.npy", score_count)
        np.save(matrices_dir / f"{probe}.npy", scores)

        edges = infer_edges(scores, adata.var_names, network_cfg)
        export_edges_tsv(edges, edges_dir / f"{probe}.tsv")

        probe_scores[probe] = scores

        pred_edges = normalize_edges(edges, alias_map)
        gene_set = set(gene_names_norm)
        pred_edges = pred_edges[
            pred_edges["source"].isin(gene_set) & pred_edges["target"].isin(gene_set)
        ].drop_duplicates()

        for ref_cfg in references_cfg:
            true_edges = load_dorothea(
                ref_cfg["path"], confidence_levels=ref_cfg.get("confidence_levels")
            )
            true_edges = normalize_edges(true_edges, alias_map)
            true_edges = true_edges[
                true_edges["source"].isin(gene_set) & true_edges["target"].isin(gene_set)
            ].drop_duplicates()

            labels = _build_label_matrix(true_edges, source_names, target_names)
            score_subset = scores[np.ix_(source_indices, target_indices)]
            pr_metrics = precision_recall_f1(pred_edges, true_edges)
            pr_at_k, rec_at_k = _precision_recall_at_k(
                score_subset, labels, int(probe_cfg.get("evaluation_top_k", 1000))
            )
            metrics_rows.append(
                {
                    "probe": probe,
                    "reference": ref_cfg.get("name", "reference"),
                    "precision": pr_metrics["precision"],
                    "recall": pr_metrics["recall"],
                    "f1": pr_metrics["f1"],
                    "aupr": aupr(score_subset.ravel(), labels.ravel()),
                    "auroc": _auroc(score_subset, labels),
                    "precision_at_k": pr_at_k,
                    "recall_at_k": rec_at_k,
                    "n_pred_edges": len(pred_edges),
                    "n_true_edges": len(true_edges),
                    "n_candidates": score_subset.size,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)

    agreement_rows = []
    probes_list = list(probe_scores.keys())
    agreement_top_k = int(probe_cfg.get("agreement_top_k", 1000))
    for i, probe_a in enumerate(probes_list):
        for probe_b in probes_list[i + 1 :]:
            scores_a = probe_scores[probe_a][np.ix_(source_indices, target_indices)]
            scores_b = probe_scores[probe_b][np.ix_(source_indices, target_indices)]
            edges_a = _top_edges_global(
                probe_scores[probe_a],
                source_names,
                target_names,
                source_indices,
                target_indices,
                agreement_top_k,
                network_cfg.remove_self,
            )
            edges_b = _top_edges_global(
                probe_scores[probe_b],
                source_names,
                target_names,
                source_indices,
                target_indices,
                agreement_top_k,
                network_cfg.remove_self,
            )
            union = edges_a | edges_b
            jaccard = float(len(edges_a & edges_b) / len(union)) if union else 0.0
            agreement_rows.append(
                {
                    "probe_a": probe_a,
                    "probe_b": probe_b,
                    "jaccard_top_k": jaccard,
                    "spearman": _spearman_corr(scores_a.ravel(), scores_b.ravel()),
                }
            )

    agreement_df = pd.DataFrame(agreement_rows)
    agreement_df.to_csv(agreement_path, sep="\t", index=False)

    recommendations = ["# Probe Benchmark Recommendations", ""]
    if not metrics_df.empty:
        mean_aupr = metrics_df.groupby("probe")["aupr"].mean()
        best_aupr_probe = mean_aupr.idxmax()
        recommendations.append(
            f"- Best mean AUPR: {best_aupr_probe} ({mean_aupr[best_aupr_probe]:.4f})"
        )
        mean_f1 = metrics_df.groupby("probe")["f1"].mean()
        best_f1_probe = mean_f1.idxmax()
        recommendations.append(
            f"- Best mean F1: {best_f1_probe} ({mean_f1[best_f1_probe]:.4f})"
        )
    if not agreement_df.empty:
        jaccard_map: Dict[str, List[float]] = {}
        for _, row in agreement_df.iterrows():
            jaccard_map.setdefault(row["probe_a"], []).append(row["jaccard_top_k"])
            jaccard_map.setdefault(row["probe_b"], []).append(row["jaccard_top_k"])
        mean_jaccard = {key: float(np.mean(vals)) for key, vals in jaccard_map.items() if vals}
        if mean_jaccard:
            best_agreement_probe = max(mean_jaccard, key=mean_jaccard.get)
            recommendations.append(
                f"- Highest agreement (Jaccard@{agreement_top_k}): {best_agreement_probe} "
                f"({mean_jaccard[best_agreement_probe]:.4f})"
            )
    recommendations.extend(
        [
            "",
            "## Notes",
            f"- Candidates: {len(source_indices)} sources x {len(target_indices)} targets",
            f"- Normalization: {probe_cfg.get('normalization', 'none')} "
            f"(axis={probe_cfg.get('normalization_axis', 'source')})",
            f"- Thresholding: top_k={network_cfg.top_k} "
            f"percentile={network_cfg.threshold_percentile}",
        ]
    )
    recommendations_path.write_text("\n".join(recommendations), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe benchmarking for scGPT")
    parser.add_argument("--config", default="configs/probe_benchmark.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--probes", nargs="*", default=None)
    args = parser.parse_args()
    run_benchmark(args.config, args.device, args.probes)


if __name__ == "__main__":
    main()
