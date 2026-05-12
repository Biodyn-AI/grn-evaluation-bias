"""Phase B2+B4 (geneformer arm): Geneformer-V1-10M attention extraction on BEELINE.

Geneformer uses rank-value encoding: per cell, genes are ranked by (counts / population_median);
the top-N ranked Ensembl IDs are tokenized and fed as a sequence. Attention from BertSelfAttention
is captured via output_attentions=True.

Outputs the same 6 aggregation variants as run_scgpt_attention.py.

Usage:
  python scripts/run_geneformer_attention.py --tag hHep [--device mps|cpu]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
PREP_ROOT = IMPL_ROOT / "outputs" / "prep"
OUT_ROOT = IMPL_ROOT / "outputs" / "geneformer"

GF_SNAPSHOT = Path("/Users/ihorkendiukhov/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/fcd26c45fc30fba1989e586bdc46bc366dda8655")
GF_MODEL_DIR = GF_SNAPSHOT / "Geneformer-V1-10M"
GF_DICT_DIR = GF_SNAPSHOT / "geneformer" / "gene_dictionaries_30m"


def _device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


class GeneformerRankDataset(Dataset):
    """Per-cell rank-value sequence; returns token_ids + gene_indices into the universe."""

    def __init__(self, X: np.ndarray, genes: list[str], gene_to_ensembl: dict[str, str],
                  ensembl_to_token: dict[str, int], ensembl_to_median: dict[str, float],
                  max_len: int = 2048, pad_token_id: int = 0):
        self.X = X
        self.genes = genes
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        # Map each universe-gene index to (token_id, median); -1 if missing.
        self.token_ids = np.full(len(genes), -1, dtype=np.int64)
        self.medians = np.full(len(genes), np.nan, dtype=np.float32)
        for i, g in enumerate(genes):
            ens = gene_to_ensembl.get(g)
            if ens is None:
                continue
            tok = ensembl_to_token.get(ens)
            med = ensembl_to_median.get(ens)
            if tok is None or med is None:
                continue
            self.token_ids[i] = tok
            self.medians[i] = med
        self.valid_mask = self.token_ids >= 0
        valid_count = int(self.valid_mask.sum())
        print(f"  Geneformer-encodable genes: {valid_count} / {len(genes)}", flush=True)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        # Per-cell normalized rank: (counts / median); zero-counts get rank 0
        norm = np.zeros_like(row, dtype=np.float32)
        nz = (row > 0) & self.valid_mask
        if not nz.any():
            return {
                "token_ids": torch.tensor([self.pad_token_id], dtype=torch.long),
                "attention_mask": torch.tensor([0], dtype=torch.long),
                "gene_indices": torch.tensor([-1], dtype=torch.long),
            }
        norm[nz] = row[nz] / self.medians[nz]
        # Top-max_len by rank
        keep = np.where(nz)[0]
        order = keep[np.argsort(-norm[keep])][: self.max_len]
        tokens = self.token_ids[order]
        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "gene_indices": torch.tensor(order, dtype=torch.long),
        }


def collate(batch):
    max_len = max(b["token_ids"].size(0) for b in batch)
    tids, masks, gix = [], [], []
    for b in batch:
        L = b["token_ids"].size(0)
        pad = max_len - L
        tids.append(torch.cat([b["token_ids"], torch.zeros(pad, dtype=torch.long)]))
        masks.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        gix.append(torch.cat([b["gene_indices"], -torch.ones(pad, dtype=torch.long)]))
    return {
        "token_ids": torch.stack(tids),
        "attention_mask": torch.stack(masks),
        "gene_indices": torch.stack(gix),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="Geneformer max_position_embeddings (V1-10M = 2048)")
    args = ap.parse_args()

    device = _device(args.device)
    print(f"[{args.tag}] device={device}", flush=True)

    # Load BEELINE prep
    h5ad = PREP_ROOT / args.tag / f"{args.tag}_raw.h5ad"
    adata = ad.read_h5ad(h5ad)
    print(f"[{args.tag}] {adata.shape[0]} cells x {adata.shape[1]} genes", flush=True)
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)  # counts
    genes = list(adata.var.index)

    # Load Geneformer dictionaries
    with open(GF_DICT_DIR / "token_dictionary_gc30M.pkl", "rb") as f:
        token_dict = pickle.load(f)
    with open(GF_DICT_DIR / "gene_name_id_dict_gc30M.pkl", "rb") as f:
        gene_name_id_dict = pickle.load(f)
    with open(GF_DICT_DIR / "gene_median_dictionary_gc30M.pkl", "rb") as f:
        gene_median_dict = pickle.load(f)
    print(f"[{args.tag}] gene_name_id size={len(gene_name_id_dict)}, token_dict size={len(token_dict)}", flush=True)
    pad_id = token_dict.get("<pad>", 0)

    dataset = GeneformerRankDataset(
        X, genes, gene_name_id_dict, token_dict, gene_median_dict,
        max_len=args.max_len, pad_token_id=pad_id,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    # Load Geneformer
    from transformers import BertModel
    model = BertModel.from_pretrained(str(GF_MODEL_DIR), attn_implementation="eager")
    model.to(device)
    model.eval()
    print(f"[{args.tag}] Geneformer loaded; hidden={model.config.hidden_size}, "
          f"L={model.config.num_hidden_layers}, H={model.config.num_attention_heads}", flush=True)

    n_genes = len(genes)
    keys = ["mean_layers_mean_heads", "max_layers_mean_heads", "mean_layers_max_heads",
             "max_layers_max_heads", "last_layer_mean_heads", "per_head_best"]
    acc = {k: np.zeros((n_genes, n_genes), dtype=np.float64) for k in keys}
    counts = np.zeros((n_genes, n_genes), dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            tok = batch["token_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gix = batch["gene_indices"]
            outputs = model(input_ids=tok, attention_mask=mask, output_attentions=True)
            attn_list = outputs.attentions  # tuple of (B,H,S,S)
            attn = torch.stack(list(attn_list), dim=0)  # (L,B,H,S,S)
            attn = attn.permute(1, 0, 2, 3, 4).contiguous()  # (B,L,H,S,S)
            attn_np = attn.cpu().float().numpy()
            gix_np = gix.numpy()
            B, L, H, S, _ = attn_np.shape
            for b in range(B):
                valid = gix_np[b] >= 0
                idxs = gix_np[b][valid]
                if len(idxs) == 0:
                    continue
                sub = attn_np[b][:, :, valid][..., valid]
                mm = sub.mean(axis=(0, 1))
                acc["mean_layers_mean_heads"][np.ix_(idxs, idxs)] += mm
                max_mean = sub.max(axis=0).mean(axis=0)
                acc["max_layers_mean_heads"][np.ix_(idxs, idxs)] += max_mean
                mean_max = sub.mean(axis=0).max(axis=0)
                acc["mean_layers_max_heads"][np.ix_(idxs, idxs)] += mean_max
                mxmx = sub.max(axis=(0, 1))
                acc["max_layers_max_heads"][np.ix_(idxs, idxs)] += mxmx
                last = sub[-1].mean(axis=0)
                acc["last_layer_mean_heads"][np.ix_(idxs, idxs)] += last
                ph = sub.max(axis=(0, 1))
                acc["per_head_best"][np.ix_(idxs, idxs)] += ph
                counts[np.ix_(idxs, idxs)] += 1
            if (bi + 1) % 25 == 0:
                el = time.time() - t0
                eta = el / (bi + 1) * (len(loader) - bi - 1)
                print(f"  batch {bi+1}/{len(loader)} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    counts_safe = np.maximum(counts, 1)
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    from prep_beeline import load_tf_list
    T = sorted(load_tf_list() & set(genes))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tf_idx = np.array([gene_to_idx[t] for t in T])

    for k in keys:
        mat = acc[k] / counts_safe
        np.fill_diagonal(mat, 0.0)
        sub = mat[tf_idx]
        src = np.repeat(np.asarray(T), len(genes))
        tgt = np.tile(np.asarray(genes), len(T))
        sc = sub.flatten().astype(np.float32)
        keep = src != tgt
        df = pd.DataFrame({"source": src[keep], "target": tgt[keep], "score": sc[keep]})
        out = out_dir / f"{k}.parquet"
        df.to_parquet(out, index=False)
        print(f"[{args.tag}] wrote {out} ({len(df)} rows)", flush=True)

    meta = {"tag": args.tag, "n_cells": int(adata.n_obs), "n_genes_in_universe": int(n_genes),
             "device": device, "max_len": args.max_len, "model": str(GF_MODEL_DIR),
             "elapsed_s": time.time() - t0}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.tag}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
