# BCB2026 Poster — Universe-Aware GRN Evaluation

This directory contains the materials for the ACM-BCB 2026 poster
**"Evaluation Protocol Choices Dominate Gene Regulatory Network
Benchmarking Outcomes for Single-Cell Foundation Models."**

It is a stand-alone, end-to-end pipeline applied to BEELINE
**hESC** and **hHep** with cell-type-specific ChIP-seq, non-specific
ChIP-seq, and STRING gold standards, comparing scGPT and Geneformer
attention against Pearson / Spearman / GRNBoost2 / random baselines
under the universe-aware protocol.

## Headline findings

- A Type-II ANOVA over 279 (dataset, gold standard, candidate set,
  method) tuples assigns **80.1%** of variance in
  log10(AUPR) to evaluation-protocol factors versus
  **0.27%** to method identity — a ~300× gap.
- Restricting the candidate set from the universe-aware
  *T × G* to the literature-standard
  *T × T_gold-targets* inflates median AUPR by
  **13–17×** without changing any underlying scores. AUPR
  tracks base rate at ρ = 0.997.
- Under universe-aware scoring against cell-type ChIP-seq, all six
  scGPT attention-aggregation variants have AUROC **below 0.5** on
  hHep (DeLong 95% CI strictly below 0.5, *p* ≪ 0.001); on hESC the
  same model reaches AUROC ≈ 0.58. The verdict flips by dataset and
  is invisible under aggregated single-number benchmarks.

## Reproducing the results

```bash
# 1. Download BEELINE data + networks (Zenodo record 3701939) into
#    bcb2026/data/beeline/  (SHA-256 of the zips is in PROVENANCE.md).
# 2. Pin the HGNC dictionary (same SHA-256 as PROVENANCE.md).

python scripts/prep_beeline.py --tag hESC
python scripts/prep_beeline.py --tag hHep

# Baselines (all on the same universe)
for m in pearson spearman random grnboost2; do
  python scripts/run_baselines.py --tag hESC --method $m --n-jobs 4
  python scripts/run_baselines.py --tag hHep --method $m --n-jobs 6
done

# Foundation-model attention (6 aggregation variants each)
python scripts/run_scgpt_attention.py    --tag hESC --device cpu --batch-size 2 --max-genes 1000
python scripts/run_scgpt_attention.py    --tag hHep --device cpu --batch-size 2 --max-genes 1000
python scripts/run_geneformer_attention.py --tag hESC --device mps --batch-size 4 --max-len 2048
python scripts/run_geneformer_attention.py --tag hHep --device mps --batch-size 4 --max-len 2048

# Universe-aware evaluation + significance + variance decomposition
python scripts/eval_universe_aware.py --tag hESC
python scripts/eval_universe_aware.py --tag hHep
python scripts/auroc_ci.py            --tag hESC --B 200 --sub-n 100000
python scripts/auroc_ci.py            --tag hHep --B 200 --sub-n 100000
python scripts/variance_decomp.py     --y log_aupr

# Paper number trace
python scripts/format_results.py
```

## Layout

```
bcb2026/
├── paper/
│   ├── poster.tex / poster.pdf                  2-page ACM-BCB poster (with author)
│   ├── poster_anonymous.tex / poster_anonymous.pdf  2-page blind-review variant
│   ├── main.tex / main.pdf                      Full 5-page version with Algorithm 1
│   └── reproducibility_appendix.tex             SHA-pinned data inventory
├── scripts/
│   ├── prep_beeline.py             Symbol mapping + universe construction
│   ├── run_baselines.py            Pearson / Spearman / GRNBoost2 / random
│   ├── run_scgpt_attention.py      6 aggregation variants from scGPT attention
│   ├── run_geneformer_attention.py 6 aggregation variants from Geneformer-V1-10M
│   ├── eval_universe_aware.py      AUPR / AUROC / top-K precision over U
│   ├── auroc_ci.py                 DeLong AUROC CI + paired AUPR bootstrap
│   ├── permutation_null.py         (Optional) 1000-permutation null AUPR
│   ├── variance_decomp.py          Type-II SS ANOVA
│   └── format_results.py           Paper-snippet generator
├── planning/
│   ├── revision_plan.md            Reviewer-issue tracker
│   └── universe_definition.md      Formal definition of U(D)
├── data/
│   └── PROVENANCE.md               SHA-256 for HGNC + BEELINE archives
└── outputs/
    └── eval/                       Per-(dataset, gold, candidate set, method) metrics
        ├── {hESC,hHep}_metrics.parquet
        ├── {hESC,hHep}_auroc_ci.parquet
        ├── {hESC,hHep}_paired_bootstrap.parquet
        └── variance_decomp_log_aupr.csv
```

## Provenance

All numeric values in the poster are emitted verbatim by these scripts
and are independently re-derivable from BEELINE Zenodo record 3701939
(SHA-256 in `data/PROVENANCE.md`) plus the pinned HGNC dictionary
(SHA-256 in `data/PROVENANCE.md`).

## License

MIT.
