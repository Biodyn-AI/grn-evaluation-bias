# GRN Evaluation Bias Protocol

A standardized evaluation framework for benchmarking gene regulatory network (GRN) inference methods, with a focus on quantifying evaluation biases introduced by identifier misalignment, candidate gating, and reference network domain mismatch.

## ACM-BCB 2026 poster

See [`bcb2026/`](bcb2026/) for the full pipeline applied to BEELINE hESC and hHep, comparing scGPT and Geneformer attention against expression baselines under the universe-aware protocol. Headline: protocol choice explains **80.1%** of AUPR variance vs **0.27%** for method identity (~300× gap); scGPT AUROC is significantly **below random** on hHep cell-type ChIP-seq under universe-aware scoring (DeLong CI, *p* ≪ 0.001).

## Motivation

GRN inference benchmarks are sensitive to methodological choices that are often left implicit: how gene identifiers are mapped across databases, which candidate edges are considered, and whether reference networks overlap with the gene universe under study. This repository provides source code, configurations, reference data, and evaluation outputs that make these choices explicit and reproducible.

## Key findings

- **Curated priors dominate learned methods.** OmniPath and DoRothEA/TRRUST priors consistently outperform attention-derived and classical GRN inference methods (GENIE3, GRNBoost2, SCENIC, PIDC, Pearson/Spearman correlation) on HPN-DREAM and BEELINE GSD benchmarks.
- **Identifier alignment matters.** Crosswalk-aware alias expansion (HGNC + Entrez + Ensembl + UniProt) eliminates missing-ID artifacts for HPN-DREAM, while BEELINE GSD retains 3 out-of-domain genes (SRY, NR0B1, UGR) that cannot be resolved for the immune gene universe.
- **Candidate gating inflates recall.** Restricting evaluation to OmniPath- or DoRothEA-gated candidate sets artificially boosts recall for methods operating within those candidate spaces.
- **Node overlap is a critical confounder.** Overlap between inferred/probe node sets and HPN-DREAM/BEELINE reference nodes is 0-2 genes for immune-subset networks, rendering overlap-driven F1/AUPR effectively out of domain.

## Repository structure

```
network_inference/           Network inference and evaluation library
  src/                       Python package (CLI, evaluation, inference, calibration, priors)
  scripts/                   Helper scripts (summary generation, report validation)
single_cell_mechinterp/      scGPT attention extraction and mechanistic interpretability
  src/                       Python package (data, eval, interpret, model, network, utils)
  scripts/                   Analysis scripts (eval bias, probes, causal interventions)
  deps/                      Conda environment and pip requirements
configs/                     YAML configuration files for evaluation runs
data/                        Symbol mapping TSVs and reference network subsets
outputs/                     Evaluation results (JSON metrics, CSV score tables, diagnostics)
REPORT.md                    Detailed technical report with methods, tables, and figures
```

### network_inference/

The `network_inference` package provides a config-driven CLI for network inference and evaluation:

```bash
# Infer a network from attention scores
python -m network_inference.src.cli infer --config configs/base.yaml

# Evaluate against reference networks (AUPR, precision, recall, F1)
python -m network_inference.src.cli evaluate-scores --config configs/score_eval_grn_baselines_immune.yaml

# Run threshold sweeps
python -m network_inference.src.cli sweep --config configs/base.yaml

# Calibrate edge scores
python -m network_inference.src.cli calibrate --config configs/base.yaml
```

Key modules:
- `src/evaluation/score_edges.py` -- Edge scoring with identifier alignment, overlap gating, and AUPR computation
- `src/evaluation/sweep.py` -- Percentile and top-k threshold sweeps
- `src/data/mapping.py` -- Gene/protein ID mapping with HGNC alias resolution
- `src/inference/candidates.py` -- Candidate edge generation with OmniPath/DoRothEA gating
- `src/calibration/` -- Isotonic and logistic calibration for edge confidence scores

### single_cell_mechinterp/

The `single_cell_mechinterp` package handles scGPT attention extraction and mechanistic interpretability probes:

Key modules:
- `src/eval/bias_protocol.py` -- Evaluation bias protocol implementation
- `src/eval/gene_symbols.py` -- Gene symbol mapping and validation
- `src/interpret/attention.py` -- Attention extraction and analysis
- `src/interpret/causal_intervention.py` -- Causal intervention framework (ablation, swap)
- `src/model/` -- scGPT model loading, hooks, and vocabulary

Key scripts:
- `scripts/run_eval_bias_protocol.py` -- Run the full evaluation bias protocol
- `scripts/generate_eval_bias_baselines.py` -- Generate GRN baselines (GENIE3, GRNBoost2, SCENIC, PIDC, correlation)
- `scripts/run_probe_benchmark.py` -- Probe-based evaluation (attention, gradients, perturbation, consensus)
- `scripts/run_causal_interventions.py` -- Causal intervention experiments

### configs/

| Config | Description |
|--------|-------------|
| `regulatory_eval.yaml` | Strict regulatory evaluation (DoRothEA/TRRUST only) |
| `score_eval_grn_baselines_immune.yaml` | GRN baseline methods vs HPN-DREAM + BEELINE + DoRothEA/TRRUST |
| `score_eval_probe_priors.yaml` | Mechanistic interpretability probes vs curated references |
| `score_eval_probe_priors_full_genes.yaml` | Same probes evaluated over the full immune gene universe |
| `score_eval_probe_priors_full_genes_crosswalk.yaml` | Full-gene evaluation with crosswalk-aware alias expansion |
| `score_eval_probe_priors_full_genes_omnipath.yaml` | Full-gene evaluation gated by OmniPath interactions |

### data/

| File | Description |
|------|-------------|
| `hpn_dream_symbol_map.tsv` | Gene symbol mapping for HPN-DREAM phosphoproteomics gold standard |
| `hpn_dream_symbol_map_crosswalk.tsv` | Extended crosswalk with Entrez/Ensembl/UniProt columns |
| `beeline_gsd_symbol_map.tsv` | Gene symbol mapping for BEELINE GSD reference network |
| `beeline_gsd_symbol_map_crosswalk.tsv` | Extended crosswalk for BEELINE GSD identifiers |
| `dorothea_trrust_union_immune.tsv` | Immune-filtered union of DoRothEA and TRRUST regulatory edges |

### outputs/

Evaluation metrics in paired CSV/JSON format:

- `score_eval_*.csv` / `score_eval_*.json` -- Precision, recall, F1, and AUPR for each method against each reference network
- `baseline_eval_hpn_beeline.*` / `grn_baseline_eval_hpn_beeline.*` -- Baseline and GRN-specific evaluations against HPN-DREAM and BEELINE
- `node_overlap_hpn_beeline_full_genes.tsv` -- Node overlap statistics between inferred networks and reference networks
- `*_missing_ids.tsv` / `*_missing_report.json` -- Identifier alignment diagnostics (unmapped genes per reference)

## Installation

```bash
conda env create -f single_cell_mechinterp/deps/environment.yml
conda activate scm
pip install -r single_cell_mechinterp/deps/requirements.txt
```

## Datasets

The evaluation framework uses the following external datasets and references:

- **[Tabula Sapiens](https://tabula-sapiens-portal.ds.czbiohub.org/)** -- Human single-cell RNA-seq atlas (immune subset for attention extraction)
- **[HPN-DREAM](https://www.synapse.org/HPN-DREAM)** -- Phosphoproteomic signaling gold standard from the DREAM challenge
- **[BEELINE GSD](https://github.com/Murali-group/Beeline)** -- Curated gene regulatory reference network
- **[DoRothEA](https://saezlab.github.io/dorothea/)** -- Curated TF-target regulon database
- **[TRRUST](https://www.grnpedia.org/trrust/)** -- Literature-curated human transcriptional regulatory network
- **[OmniPath](https://omnipathdb.org/)** -- Comprehensive signaling and regulatory interaction database

## Methods evaluated

| Category | Methods |
|----------|---------|
| Attention-derived | scGPT attention (raw, OmniPath-gated, DoRothEA-gated variants) |
| Mechanistic interpretability probes | Attention, gradient x input, integrated gradients, perturbation, consensus |
| Classical GRN inference | GENIE3, GRNBoost2, SCENIC (GRNBoost2 + pruned), PIDC |
| Correlation baselines | Pearson, Spearman |
| Curated priors | OmniPath, TRRUST, DoRothEA/TRRUST union |
| Random baseline | Uniform random edge selection |


## License

This project is released under the [MIT License](LICENSE).
