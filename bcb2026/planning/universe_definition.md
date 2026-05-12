# §3.1 Formal definition of the universe-aware protocol

This document is the canonical definition that will be inlined verbatim into the paper. It exists to fix the central reviewer issue (xf7H X.2, X.3, X.4 and N1DC N1.2): the universe-aware protocol was named but not defined in the submitted abstract.

## Setup

Let a dataset `D` consist of a single-cell expression matrix `X ∈ R^{n×p}` with `n` cells and `p` raw genes, plus a cell-type label. Let `S(g)` denote the HGNC-approved symbol of gene `g` under the pinned mapping policy (§3.2).

## The gene set `G(D)`

After the BEELINE standard pipeline (quality control + log-normalization + variable-gene retention), let
```
G(D) = { S(g) : gene g passes QC in D, and S(g) is HGNC-approved }
```

We retain a gene only if (a) it survives BEELINE's default QC, (b) it maps to an HGNC-approved symbol after the pinned policy, and (c) it is detected in ≥ 3 cells. Self-pairs are excluded later.

## The TF set `T`

`T` is the union of the BEELINE-provided human TF list (`human-tfs.csv`, 1,563 entries) and the Lambert et al. 2018 list, restricted to HGNC-approved symbols and intersected with `G(D)`:

```
T(D) = (T_BEELINE ∪ T_Lambert) ∩ G(D)
```

`T(D)` is what we treat as "the set of biologically plausible regulators in this dataset." Reporting both `T_BEELINE` alone and the union is part of the robustness check.

## The universe `U(D)` — universe-aware candidate set

```
U(D) = { (s, t) : s ∈ T(D), t ∈ G(D), s ≠ t }
```

Equivalently, `U(D) = T(D) × (G(D) \\ {self})`. Reported with the paper: |G(D)|, |T(D)|, |U(D)|, |E(D)| (number of gold-standard edges in `U(D)`), and base rate `b(D) = |E(D)| / |U(D)|`.

## Comparison protocols (what literature uses)

For each evaluation we report AUPR/AUROC under four candidate sets to expose the inflation effect:

| Name | Candidate set | Typical |U| (hESC scale) |
|---|---|---|
| `all_pairs` | `(G \ {self}) × (G \ {self})` | ~3e8 |
| `tf_sources` (a.k.a. universe-aware) | `T × (G \ {self})` | ~3e7 |
| `tf_sources_targets` (current convention, biased) | `T × T_targets`, where `T_targets ⊆ T` are TFs that appear as targets in the gold standard | ~2e6 |
| `gold_recovered_targets_only` (most extreme, sometimes used implicitly) | edges restricted to (s,t) where t is a known target in the gold standard | ~5e5 |

The progression `all_pairs → tf_sources → tf_sources_targets → gold_recovered_targets_only` raises base rate monotonically by roughly two orders of magnitude per step.

**The paper's positive recommendation:** report AUPR under `tf_sources` (universe-aware) as the primary number. Report `all_pairs` and `tf_sources_targets` for transparency. Do not report restricted protocols without disclosing the candidate set explicitly.

## "Below random" — precise statement

Given an evaluation tuple `(D, candidate_set, method)`, denote the AUPR by `AUPR(D, c, m)`. We say a method `m` is **below random** on `(D, c)` when

```
AUPR(D, c, m) < q_5( {AUPR(D, c, π) : π ∈ Π_K} )
```

where `Π_K = { K independent uniform random permutations of edge scores over U(D) }` with `K = 1000`, and `q_5` is the 5th-percentile. We also report the empirical p-value `p = (#{π : AUPR(D,c,π) ≥ AUPR(D,c,m)} + 1) / (K + 1)`.

In addition we report `Δ_m = AUPR(D, c, m) − mean( AUPR(D, c, π) )` and its 95% paired-bootstrap CI over 2,000 resamples of `U(D)`. A "below random" claim is published only when both: (i) `p < 0.05`, and (ii) the upper bound of the CI of `Δ_m` is `< 0`.

This makes every "below random" headline statistically falsifiable, which addresses reviewer xf7H X.4 directly.

## What the abstract will report

For each dataset `D ∈ {hESC, hHep}`:
- |G(D)|, |T(D)|, |U(D)|, |E(D)|, b(D) — five numbers.
- AUPR/AUROC under the four candidate sets for scGPT + Geneformer + 4 baselines + random — single primary table.
- p-value of "below random" for scGPT and Geneformer under `tf_sources`.
- Inflation factor: `AUPR(D, tf_sources_targets, m) / AUPR(D, all_pairs, m)` for `m = scGPT`.

## What changes vs. the workshop draft

- Workshop draft used Tabula Sapiens kidney/lung/immune. This revision uses BEELINE hESC + hHep so that the experiments match the abstract's specific cell-type claims and the BEELINE benchmark conventions cited in the literature.
- Workshop's TF list came implicitly from gold standards; this revision specifies `T(D)` explicitly via the union approach.
- Workshop reported `tf_sources_targets` as the headline metric; this revision designates `tf_sources` (universe-aware) as the headline and reports the rest as comparison.
