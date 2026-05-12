# BCB2026 Revision Plan — "Evaluation Protocol Choices Dominate GRN Benchmarking Outcomes"

Source paper: `/Users/ihorkendiukhov/biodyn-work/BCB2026_poster_eval_bias.tex`
Source draft (background, Tabula Sapiens basis): `reports/evaluation_bias_protocol/workshop/eval_bias_paper_draft.md`

## 0. Decisions locked in (2026-05-11)

1. **Dataset basis:** Run real BEELINE hESC/hHep experiments. The submitted abstract makes specific claims about BEELINE hESC/hHep that no artifact in the repo currently backs. The Tabula Sapiens draft is background only.
2. **Scope:** Full ACM-BCB short paper (6–8 pages) — methods, figures, tables, limitations. The current submission is essentially abstract + 4 references and that is the structural cause of "lacks fundamental details."
3. **Added experiments:** all four — permutation null for "below random," Geneformer as a second foundation model, attention-aggregation ablation, full-budget scGPT.

---

## 1. Reviewer issues — material vs minor

### Reviewer N1DC (rating 5)
| # | Issue | Class | Where addressed |
|---|---|---|---|
| N1.1 | Abstract hard to read | Material | §5 paper rewrite |
| N1.2 | "Universe-aware" protocol introduced but not explained | Material | §3 formal definition, §5 methods section |
| N1.3 | Lacks fundamental details (methodology poorly reported) | Material | §5 full methods section |

### Reviewer xf7H (rating 4)
| # | Issue | Class | Where addressed |
|---|---|---|---|
| X.1 | Avoid acronym GRN in title | Minor | §5 title revision |
| X.2 | Technical definition of "all biologically plausible TF–target pairs" missing | Material | §3 universe formalization |
| X.3 | How was the universe defined? How many edges considered? | Material | §3 universe + §4 explicit counts |
| X.4 | "Below random" claim not verifiable without universe definition | Material | §3 universe + §4.1 permutation null |
| X.5 | Methodology poorly reported | Material | §5 methods section |

---

## 2. Additional issues I identified (review pass)

### Material (must fix)
| # | Issue | Why it matters | Where addressed |
|---|---|---|---|
| A.1 | Submitted abstract claims **BEELINE hESC/hHep** but no artifacts exist; the only completed work uses **Tabula Sapiens kidney/lung/immune**. The numbers ρ=0.99, 5–12% HGNC shift, scGPT-below-random on hESC/hHep are not currently backed by any pipeline output I could locate. | If experiments cited in the abstract were never run, every numeric claim is unsupported — this is the deepest version of "lacks fundamental details." | §4 experiments — run BEELINE hESC/hHep end-to-end |
| A.2 | Only one foundation model (scGPT) evaluated, but the title and abstract generalize to "single-cell foundation models" (plural) | Generalization claim unsupported with n=1 model | §4.2 add Geneformer |
| A.3 | scGPT runs use reduced budget (`max_genes=1000`, `max_cells=2000`, no flash-attn) | Reviewers will read this as "unfair scGPT" and discount the negative result | §4.3 full-budget rerun |
| A.4 | "Raw multi-head attention aggregation does not recover regulatory structure" stated, but only one aggregation (max-across-heads, mean-across-layers) is tested | Conclusion stronger than the evidence | §4.4 aggregation ablation |
| A.5 | "scGPT below random" claim is a point-estimate comparison; the gap (e.g., 0.00499 vs 0.0117 on kidney) lies within the bootstrap CI width (7.86e-03 for TF-source-target) | Headline "below random" is not statistically established | §4.1 permutation null + paired bootstrap |
| A.6 | Baselines GENIE3/GRNBoost2 trained on subset (1k targets, 500 regulators, 2k cells) while scGPT runs on the same subset — but the comparison was *not* explicitly matched. The asymmetry is unclear in the abstract | Apples-to-apples comparison must be guaranteed and stated | §5 methods, equal candidate set and equal gene universe across all methods |
| A.7 | AUROC values in the appendix are all ≈ 0.50 (no method exceeds 0.51) — a striking signal that the entire benchmark sits near chance under TF-target-target | Strengthens the paper if surfaced; weakens it if hidden | §5 results, surface in headline figure |
| A.8 | Symbol mapping ambiguity stated as "5–12%" in TeX abstract vs. "4.1e-04 mean edge coverage shift across policies" in draft | Number must be defined precisely and traceable to a table | §3.2 mapping definition + §4 mapping counts table |
| A.9 | No comparison to the published scGPT numbers — the paper critiques benchmarks but doesn't show the original claim being deflated side-by-side | Needed to make the headline land | §4.5 reproduce-then-reframe table |
| A.10 | No formal statement of the protocol (algorithm/pseudocode) | Reviewers asked for protocol; words alone are ambiguous | §5 add Algorithm 1 |
| A.11 | "Hypothesis generator not predictor" framing only in passing; the constructive contribution (six recommendations) is buried in the abstract | Constructive contribution should be visible and tested | §5 dedicated "Recommendations" section + checklist appendix |

### Minor (should fix)
| # | Issue | Where addressed |
|---|---|---|
| B.1 | Title contains acronym GRN | §5 title revision |
| B.2 | Bibliography has only 4 entries, missing BEELINE paper, DoRothEA, TRRUST, GENIE3 properly, GRNBoost2, Geneformer | §5 expand bibliography |
| B.3 | Abstract is one giant paragraph; structure would help readability | §5 abstract rewrite (problem→method→finding→recommendation) |
| B.4 | "candidate-set restriction" vs "universe-aware" terminology used interchangeably — pick one and define once | §3 fix terminology |
| B.5 | Recommendation list (6 items) is dense; convert to numbered list with one sentence each | §5 recommendations section |
| B.6 | Reproducibility — no DOI/Zenodo for code, no environment lockfile mentioned | §5 add reproducibility appendix |

---

## 3. Formal definitions to add to the paper

### 3.1 The universe (xf7H X.2, X.3)
Define explicitly:

> Let **G** be the set of genes expressed in the processed scRNA-seq dataset after the standard QC pipeline (min 200 genes/cell, min 3 cells/gene, HVG=5000). Let **T ⊆ G** be the set of human transcription factors (defined by the union of TF lists from Lambert et al. 2018, AnimalTFDB, and DoRothEA). The **universe-aware candidate set** is **U = T × (G \ {self})**, i.e., every ordered pair (TF, target) where TF ∈ T and target is any expressed gene other than the TF itself.

Report:
- |G|, |T|, |U| for hESC and hHep, with the exact gene list versioned.
- The three competing candidate sets in the literature: all-pairs (|G|²), TF-source (T × G), TF-source-target (T × T_target) and how each inflates base rate vs U.

### 3.2 Symbol mapping policy
Pin the HGNC release date and version (e.g., HGNC dump from 2025-12-XX). Quantify mapping shift as:
- |Δedges| = number of edges that change status between two policies / total edges
- Report as a percentage and ensure the abstract's "5–12%" claim is traceable to this table.

### 3.3 "Below random" statistic
- **Null:** edge scores permuted uniformly over U.
- **Test statistic:** AUPR.
- **Decision rule:** scGPT AUPR is "below random" iff observed AUPR < 5th percentile of null AUPR distribution (1000 permutations).
- Report p-value, effect size, and 95% paired bootstrap CI of (scGPT_AUPR − random_AUPR).

---

## 4. New experiments to run

### 4.1 Real BEELINE hESC/hHep experiments (issue A.1, X.3, X.4)
- Datasets: BEELINE-provided hESC and hHep scRNA-seq + matching cell-type-specific gold standards (STRING + ChIP-seq + Loss-of-Function from BEELINE).
- Replicate the full evaluation pipeline (mapping → universe → scGPT attention → baselines → metrics) on these two datasets.
- Tables: per-dataset coverage, candidate-set sizes (|U|), base rates, AUPR/AUROC with bootstrap CIs, top-K precision/recall.
- Acceptance test: numbers in the rewritten abstract must point to specific rows in the released CSV (not memory; cross-check per `feedback_review_methodology.md`).

### 4.2 Add Geneformer as a second foundation model (issue A.2)
- Use the published Geneformer checkpoint; extract attention with comparable aggregation.
- Score against same universe and same gold standards.
- Add Geneformer rows to all method-comparison tables and figures.
- Plan: if Geneformer ≈ scGPT (both below random), strengthens "single-cell foundation models" plural claim; if Geneformer ≫ scGPT, retitle to "scGPT-style attention" or report both.

### 4.3 Full-budget scGPT rerun (issue A.3)
- Remove `max_genes=1000`, `max_cells=2000` constraints.
- Run on GPU with flash-attn enabled.
- Compare reduced-budget vs full-budget AUPR; if gap < CI width, defend reduced runs; if gap is material, use full-budget as the headline number.

### 4.4 Attention aggregation ablation (issue A.4)
- Variants: (a) max-across-heads × mean-across-layers (current), (b) mean × mean, (c) per-head best, (d) last-layer only, (e) attention rollout, (f) TF-target restricted attention only.
- Report AUPR for each variant on hESC + hHep under universe-aware protocol.
- Conclusion: "no aggregation choice we tested exceeds expression baselines on universe-aware AUPR."

### 4.5 Reproduce-then-reframe (issue A.9)
- Reproduce the original scGPT GRN-inference numbers under the *original* (TF-source-target) protocol — show they match the published values.
- Then move to universe-aware and show the drop.
- One side-by-side table makes the headline land: "Original protocol AUPR = 0.027; universe-aware AUPR = 0.0046; below random AUPR = 0.012."

### 4.6 Permutation null + paired bootstrap (issue A.5)
- 1000 permutations per (dataset, candidate-set, method).
- Paired bootstrap of (model − random) differences at 2000 resamples; report 95% CI.
- This converts every "below random" statement into a statistically defensible one.

### 4.7 ChIP-seq overlap sanity check (xf7H verifiability)
- Top-K (K=50, 100, 500, 1000) attention edges per dataset; compute fraction overlapping ENCODE ChIP-seq peaks for the matching TF.
- This grounds the "near-zero overlap" abstract claim with concrete numbers.

### 4.8 Effect-size decomposition (new framing experiment)
- ANOVA-style decomposition of AUPR variance: (candidate set | mapping policy | method | gold standard | tissue).
- Headline: protocol choices explain X% of variance; model identity explains Y%.
- This directly quantifies "protocol dominates outcome" in the title.

---

## 5. Paper rewrite — section-by-section deltas

### 5.1 Title (B.1)
- Current: "Evaluation Protocol Choices Dominate GRN Benchmarking Outcomes for Single-Cell Foundation Models"
- Revised: **"Evaluation Protocol Choices Dominate Gene Regulatory Network Benchmarking Outcomes for Single-Cell Foundation Models"** — spell out the only acronym.
- Alternative if word-count tight: "Evaluation Protocols, Not Models, Dominate Benchmark Outcomes for Single-Cell Foundation Models in Gene Regulatory Network Inference."

### 5.2 Abstract (N1.1, B.3)
Structure as four short paragraphs:
1. **Problem.** Foundation models report competitive GRN performance; we ask whether the metric reflects model quality or evaluation choice.
2. **Method.** Universe-aware protocol: define U = T × (G\{self}); use BEELINE hESC/hHep + ENCODE ChIP-seq; compare scGPT + Geneformer + 4 baselines under matched universes.
3. **Findings.** Three quantified results with concrete numbers traceable to tables.
4. **Recommendations.** One sentence pointing to the six-item checklist.

### 5.3 Sections to add to the body
- **§1 Introduction** (½ page): set up the empirical gap.
- **§2 Related work** (½ page): BEELINE, scGPT GRN claim, prior evaluation critiques (Pratapa, Saelens).
- **§3 Methods** (1.5 pages):
  - §3.1 Datasets and universe (formal definition box).
  - §3.2 Symbol mapping (pinned HGNC version, policies).
  - §3.3 Predictions: scGPT, Geneformer (aggregation choices), GENIE3, GRNBoost2, Pearson, Spearman, random.
  - §3.4 Metrics (AUPR, AUROC, top-K, base rate; bootstrap and permutation procedures).
  - §3.5 **Algorithm 1: Universe-aware GRN evaluation** (pseudocode).
- **§4 Results** (2 pages):
  - §4.1 Candidate set dominates AUPR (figure + table).
  - §4.2 Symbol mapping shifts coverage (table).
  - §4.3 Foundation models below baselines + permutation-test "below random" panel.
  - §4.4 ChIP-seq overlap is near zero.
  - §4.5 Aggregation ablation does not change the conclusion.
  - §4.6 Variance decomposition (protocol ≫ model).
- **§5 Discussion + Limitations** (½ page): explicitly bound the claim ("does not preclude…"), enumerate threats to validity.
- **§6 Recommendations** (½ page): six-item numbered checklist with one sentence each.
- **Appendix:** reproducibility checklist, environment lockfile pointer, full per-tissue/per-gold-standard tables.

### 5.4 Bibliography expansion (B.2)
Must cite: scGPT (Cui 2024), Geneformer (Theodoris 2023), BEELINE (Pratapa 2020), DoRothEA (Garcia-Alonso 2019), TRRUST (Han 2018), GENIE3 (Huynh-Thu 2010), GRNBoost2 (Moerman 2019), HGNC (Tweedie 2021), ENCODE (Davis 2018), Tabula Sapiens (Quake 2022), Lambert TF list (Lambert 2018), plus 2–3 prior critique papers on GRN evaluation.

---

## 6. Execution order and dependencies

```
Phase A — Reconcile (blocks everything)
  A1. Confirm BEELINE hESC/hHep data + gold standards are downloadable
  A2. Decide universe definition (T list source, G filter rules) — write up §3.1
  A3. Pin HGNC dump version

Phase B — New experiments (run in parallel after A)
  B1. Full-budget scGPT on hESC + hHep
  B2. Geneformer on hESC + hHep
  B3. Baselines (GENIE3, GRNBoost2, Pearson, Spearman, random) on hESC + hHep — equal universe
  B4. Aggregation ablation (uses cached attention from B1, B2)
  B5. Permutation null + paired bootstrap (post-hoc on scored outputs)
  B6. ChIP-seq overlap (independent, can run anytime after A)
  B7. Variance decomposition ANOVA (post-hoc)

Phase C — Paper rewrite (blocks on B numbers being final)
  C1. Title + abstract rewrite
  C2. Methods section (formal definitions, Algorithm 1)
  C3. Results figures + tables
  C4. Discussion + recommendations + appendix
  C5. Bibliography expansion

Phase D — Internal review passes (per `feedback_review_methodology.md`)
  D1. Reviewer-style pass 1: numbers ↔ CSV cross-check
  D2. Reviewer-style pass 2: overclaim scan
  D3. Reviewer-style pass 3: figure/table sanity
  D4. Continue passes until user signals stop
```

---

## 7. Acceptance tests before resubmission

- [ ] Every numeric claim in the abstract maps to a specific row+column in a released CSV under `subproject_02_evaluation_bias_protocol/implementation/.../outputs/` — cross-checked, not from memory.
- [ ] "Universe-aware" is defined as a single bolded sentence in §3.1 with cardinality |U| reported for each dataset.
- [ ] "Below random" claims carry a p-value and 95% paired bootstrap CI.
- [ ] Title contains zero acronyms.
- [ ] Foundation-model plural claim is supported by ≥2 models or scoped to scGPT-style attention.
- [ ] scGPT was run at full budget OR a sensitivity panel shows reduced-budget AUPR is within CI of full-budget AUPR.
- [ ] Algorithm 1 (pseudocode) is present.
- [ ] Six recommendations are explicit, numbered, and each has at least one citation or experimental backing.
- [ ] Bibliography ≥ 15 entries.

---

## 8. Open questions for the user (do not block initial work)

1. Compute budget — is GPU available for full-budget scGPT + Geneformer attention extraction? If not, document the reduced-budget sensitivity check as the substitute (issue A.3).
2. Should the recommendation checklist also be released as a standalone reproducibility tool (separate Zenodo/GitHub asset) — strengthens the constructive contribution claim.
3. Camera-ready vs. major-revision route: BCB2026 may allow direct camera-ready; if not, the revised manuscript should be ready for a different venue (TCBB or Biosystems trees already exist in the repo).
