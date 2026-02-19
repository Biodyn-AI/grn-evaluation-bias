from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def load_head_layer_scores(scores_path: str | Path, counts_path: str | Path) -> np.ndarray:
    score_sum = np.load(scores_path, mmap_mode="r")
    score_count = np.load(counts_path, mmap_mode="r")

    ensure_mechinterp_path()
    from src.interpret.attention import finalize_attention_scores

    return finalize_attention_scores(score_sum, score_count)


def edge_head_layer_evidence(
    edges: pd.DataFrame,
    gene_names: np.ndarray,
    head_layer_scores: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    if head_layer_scores.ndim != 4:
        raise ValueError("Head-layer attention scores must be 4D (layers, heads, genes, genes)")

    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    records: List[Dict[str, object]] = []

    for _, row in edges.iterrows():
        src = row["source"]
        tgt = row["target"]
        src_idx = gene_to_idx.get(src)
        tgt_idx = gene_to_idx.get(tgt)
        if src_idx is None or tgt_idx is None:
            continue
        scores = head_layer_scores[:, :, src_idx, tgt_idx]
        flat = scores.reshape(-1)
        if flat.size == 0:
            continue
        top_idx = np.argpartition(-flat, min(top_k, flat.size) - 1)[:top_k]
        top_sorted = top_idx[np.argsort(-flat[top_idx])]

        evidence = []
        for idx in top_sorted:
            layer = int(idx // scores.shape[1])
            head = int(idx % scores.shape[1])
            evidence.append({"layer": layer, "head": head, "score": float(flat[idx])})

        records.append(
            {
                "source": src,
                "target": tgt,
                "edge_score": float(row.get("score", np.nan)),
                "max_head_layer_score": float(flat[top_sorted[0]]),
                "top_head_layers": json.dumps(evidence),
            }
        )

    return pd.DataFrame.from_records(records)


def write_head_layer_evidence(
    edges_path: str | Path,
    output_path: str | Path,
    head_layer_scores_path: str | Path,
    head_layer_counts_path: str | Path,
    gene_names: np.ndarray,
    top_k: int = 5,
    min_score: float | None = None,
    top_n: int | None = None,
) -> int:
    edges = pd.read_csv(edges_path, sep="\t")
    if min_score is not None and "score" in edges.columns:
        edges = edges[edges["score"] >= float(min_score)]
    if top_n is not None and "score" in edges.columns:
        edges = edges.sort_values("score", ascending=False).head(int(top_n))
    scores = load_head_layer_scores(head_layer_scores_path, head_layer_counts_path)
    evidence = edge_head_layer_evidence(edges, gene_names, scores, top_k=top_k)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output_path, sep="\t", index=False)
    return len(evidence)
