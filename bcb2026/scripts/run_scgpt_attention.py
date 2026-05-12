"""Phase B1+B4: scGPT attention extraction on BEELINE hESC/hHep with multiple aggregation variants.

Reuses single_cell_mechinterp infra (model loader, dataset, attention.py) but works against
the BEELINE-derived prep h5ad directly. Stores per-layer per-head accumulated attention
so that aggregation ablation (B4) can be done post-hoc without rerunning the model.

Variants computed and written (one parquet each):
  mean_layers_mean_heads     (current default — for headline)
  max_layers_mean_heads
  mean_layers_max_heads      (abstract's stated choice)
  max_layers_max_heads
  last_layer_mean_heads
  per_head_best              (max over (layer, head) of attention[i,j])

Usage:
  python scripts/run_scgpt_attention.py --tag hESC [--max-cells 1500] [--device mps|cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
PREP_ROOT = IMPL_ROOT / "outputs" / "prep"
OUT_ROOT = IMPL_ROOT / "outputs" / "scgpt"
SCMI_ROOT = Path("/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp")
sys.path.insert(0, str(SCMI_ROOT))

from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt  # noqa
from src.model.scgpt_loader import load_scgpt_model  # noqa
from src.model.vocab import load_vocab  # noqa
from src.model.wrapper import ScGPTWrapper  # noqa
from src.utils.torch_utils import move_to_device  # noqa


def _device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def _accumulate(attn_4d: torch.Tensor, gene_indices: torch.Tensor,
                acc: dict[str, np.ndarray], counts: np.ndarray, n_genes: int,
                ignore_index: int = -1) -> None:
    """Update accumulators given attn_4d shape (batch, layers, heads, seq, seq)."""
    attn_np = attn_4d.detach().cpu().float().numpy()
    idx_np = gene_indices.detach().cpu().numpy()
    B, L, H, S, _ = attn_np.shape
    for b in range(B):
        valid = idx_np[b] >= 0
        idxs = idx_np[b][valid]
        if len(idxs) == 0:
            continue
        sub = attn_np[b][:, :, valid][..., valid]  # (L, H, k, k) where k=len(idxs)
        # mean-mean
        mm = sub.mean(axis=(0, 1))
        acc["mean_layers_mean_heads"][np.ix_(idxs, idxs)] += mm
        # max-mean (max over layers first, mean over heads)
        max_mean = sub.max(axis=0).mean(axis=0)
        acc["max_layers_mean_heads"][np.ix_(idxs, idxs)] += max_mean
        # mean-max (mean over layers first, max over heads)  <-- abstract's claim
        mean_max = sub.mean(axis=0).max(axis=0)
        acc["mean_layers_max_heads"][np.ix_(idxs, idxs)] += mean_max
        # max-max (max over both)
        mxmx = sub.max(axis=(0, 1))
        acc["max_layers_max_heads"][np.ix_(idxs, idxs)] += mxmx
        # last-layer mean over heads
        last = sub[-1].mean(axis=0)
        acc["last_layer_mean_heads"][np.ix_(idxs, idxs)] += last
        # per-head best
        ph = sub.max(axis=(0, 1))
        acc["per_head_best"][np.ix_(idxs, idxs)] += ph
        counts[np.ix_(idxs, idxs)] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["hESC", "hHep"])
    ap.add_argument("--max-cells", type=int, default=-1,
                    help="Stratified subsample of cells (default -1 = all)")
    ap.add_argument("--max-genes", type=int, default=1200,
                    help="Per-cell max genes (scGPT max_seq_len is 1200; default 1200)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    device = _device(args.device)
    print(f"[{args.tag}] device={device}", flush=True)

    # Load data
    h5ad = PREP_ROOT / args.tag / f"{args.tag}_raw.h5ad"
    adata = ad.read_h5ad(h5ad)
    print(f"[{args.tag}] loaded {adata.shape[0]} cells x {adata.shape[1]} genes from {h5ad.name}", flush=True)
    # scGPT expects log1p-normalized values, not counts
    if "log1p" in adata.layers:
        adata.X = adata.layers["log1p"]
        print(f"[{args.tag}] switched X to log1p layer", flush=True)

    # Subsample cells
    rng = np.random.default_rng(args.seed)
    if args.max_cells > 0 and adata.n_obs > args.max_cells:
        idx = rng.choice(adata.n_obs, size=args.max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        print(f"[{args.tag}] subsampled to {adata.n_obs} cells", flush=True)

    # Load scGPT vocab + model
    vocab_path = SCMI_ROOT / "external" / "scGPT_checkpoints" / "whole-human" / "vocab.json"
    ckpt_path = SCMI_ROOT / "external" / "scGPT_checkpoints" / "whole-human" / "best_model.pt"
    args_path = SCMI_ROOT / "external" / "scGPT_checkpoints" / "whole-human" / "args.json"
    vocab = load_vocab(str(vocab_path))

    # Filter to vocab-present genes
    present = [g for g in adata.var.index if g in vocab.gene_to_id]
    print(f"[{args.tag}] genes in scGPT vocab: {len(present)} / {adata.n_vars}", flush=True)
    adata = adata[:, present].copy()

    # Build dataset
    pad_id = vocab.gene_to_id.get("<pad>") or vocab.gene_to_id.get("[PAD]") or 0
    cfg = ScGPTDatasetConfig(
        max_genes=(adata.n_vars if args.max_genes < 0 else args.max_genes),
        include_zero=False,
        sort_by_expression=True,
        pad_token_id=pad_id,
        cls_token_id=None,
    )
    dataset = ScGPTDataset(adata, vocab.gene_to_id, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_scgpt, num_workers=0)

    # Load model
    with open(args_path) as f:
        scgpt_args = json.load(f)
    from src.cli import _build_scgpt_model_args  # reuse
    model_args = _build_scgpt_model_args(scgpt_args, vocab)
    # The entrypoint must NOT use fast/flash attention since we need output_attentions
    model_args["use_fast_transformer"] = False
    repo_path = SCMI_ROOT / "external" / "scGPT"
    model, missing, unexpected = load_scgpt_model(
        entrypoint="scgpt.model.TransformerModel",
        repo_path=str(repo_path),
        checkpoint_path=str(ckpt_path),
        device=device,
        model_args=model_args,
        prefix_to_strip=None,
    )
    print(f"[{args.tag}] model loaded; missing={len(missing) if missing else 0}, unexpected={len(unexpected) if unexpected else 0}", flush=True)
    model.to(device)
    wrapper = ScGPTWrapper(model, forward_key_map={
        "gene_ids": "src",
        "gene_values": "values",
        "src_key_padding_mask": "src_key_padding_mask",
    })
    wrapper.eval()
    print(f"[{args.tag}] model loaded", flush=True)

    # Accumulators
    n_genes = adata.n_vars
    keys = ["mean_layers_mean_heads", "max_layers_mean_heads", "mean_layers_max_heads",
             "max_layers_max_heads", "last_layer_mean_heads", "per_head_best"]
    acc = {k: np.zeros((n_genes, n_genes), dtype=np.float64) for k in keys}
    counts = np.zeros((n_genes, n_genes), dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            _, attentions = wrapper.forward_with_attentions(batch)
            if attentions is None:
                raise RuntimeError("scGPT did not return attentions")
            # attentions: list[Tensor(B,H,S,S)] or Tensor(L,B,H,S,S)
            if isinstance(attentions, torch.Tensor):
                attn = attentions
            else:
                attn = torch.stack(list(attentions), dim=0)  # (L,B,H,S,S)
            # Move (L,B,H,S,S) -> (B,L,H,S,S)
            if attn.dim() == 5:
                attn = attn.permute(1, 0, 2, 3, 4).contiguous()
            else:
                raise ValueError(f"unexpected attn shape {attn.shape}")
            _accumulate(attn, batch["gene_indices"], acc, counts, n_genes)
            if (bi + 1) % 5 == 0 or bi < 3:
                el = time.time() - t0
                eta = el / (bi + 1) * (len(loader) - bi - 1)
                print(f"  batch {bi+1}/{len(loader)} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    counts_safe = np.maximum(counts, 1)
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = list(adata.var.index)
    from prep_beeline import load_tf_list
    T = sorted(load_tf_list() & set(genes))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tf_idx = np.array([gene_to_idx[t] for t in T])

    for k in keys:
        mat = acc[k] / counts_safe  # mean per pair, where pair was observed
        # zero diagonal
        np.fill_diagonal(mat, 0.0)
        # edges from T x (G \ self)
        sub = mat[tf_idx]  # (|T|, |G|)
        src = np.repeat(np.asarray(T), len(genes))
        tgt = np.tile(np.asarray(genes), len(T))
        sc = sub.flatten().astype(np.float32)
        keep = src != tgt
        df = pd.DataFrame({"source": src[keep], "target": tgt[keep], "score": sc[keep]})
        out = out_dir / f"{k}.parquet"
        df.to_parquet(out, index=False)
        print(f"[{args.tag}] wrote {out} ({len(df)} rows)", flush=True)

    # Save metadata
    meta = {
        "tag": args.tag,
        "n_cells_used": int(adata.n_obs),
        "n_genes_used": int(adata.n_vars),
        "device": device,
        "max_genes": cfg.max_genes,
        "scgpt_checkpoint": str(ckpt_path),
        "elapsed_s": time.time() - t0,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.tag}] done in {time.time()-t0:.0f}s; meta saved", flush=True)


if __name__ == "__main__":
    main()
