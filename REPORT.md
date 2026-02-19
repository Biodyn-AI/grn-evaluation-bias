# scGPT Signaling and Regulatory Network Inference Report

## Abstract
We evaluate scGPT attention-derived gene interaction networks for signaling and regulatory inference on
Tabula Sapiens immune subset data. Networks are constrained by curated priors (OmniPath and TRRUST) and
benchmarked against HPN-DREAM and BEELINE GSD references. We also report perturbation validation using
scPerturb datasets (Adamson, Dixit, Shifrut) with causal intervention scoring and forced gene coverage.
Curated priors remain stronger baselines on F1, while perturbation-derived causal runs provide
sanity-check evidence under constrained candidate sets.

## Datasets and priors
- Tabula Sapiens immune subset (attention extraction).
- HPN-DREAM phosphoproteomic gold standard.
- BEELINE GSD reference network.
- TRRUST / DoRothEA regulatory priors (HGNC alias normalization).
- OmniPath interactions and intercell annotations.
- scPerturb: Adamson (K562), Dixit (K562 TFs 13d and 7d), Shifrut (T cell CRISPR).

## Methods summary
- Attention extraction from scGPT with head/layer reduction and aggregated attention matrices.
- Network inference via per-source top-k or percentile thresholds.
- Candidate gating using OmniPath, intercell roles, and DoRothEA/TRRUST.
- Evaluation against HPN-DREAM and BEELINE with symbol mapping and HGNC alias normalization.
- Calibration with isotonic/logistic models on DoRothEA labels.
- Causal interventions (ablation and swap) to score TF-target pairs with optional forced gene coverage.

## Experiments and configurations
Key configs and scripts:
- Immune attention: `single_cell_mechinterp/configs/immune_subset_hpn.yaml`
- Scaled attention: `single_cell_mechinterp/configs/immune_subset_hpn_scaled_1200x420.yaml`
- OmniPath + DoRothEA intersection: `network_inference/configs/omnipath_signaling_dorothea_intersection_immune_hpn.yaml`
- OmniPath + DoRothEA union: `network_inference/configs/omnipath_signaling_dorothea_union_immune_hpn.yaml`
- Intercell union (sources): `network_inference/configs/omnipath_signaling_intercell_union_sources_immune_hpn.yaml`
- GRN baselines: `single_cell_mechinterp/configs/eval_bias_baselines_immune_subset_full.yaml`
- SCENIC pruning: `single_cell_mechinterp/scripts/run_scenic_pruning.py`
- Dixit causal full run: `single_cell_mechinterp/configs/causal_intervention_dixit_targets_large.yaml`
- Dixit perturbation validation (full): `single_cell_mechinterp/configs/perturbation_validation_dixit_causal_dixit_targets_large.yaml`
- Shifrut perturbation validation (tiny): `single_cell_mechinterp/configs/perturbation_validation_shifrut_causal_shifrut_targets_tiny_neg.yaml`

## Result highlights
- Best attention-derived HPN-DREAM F1: OmniPath + DoRothEA intersection (F1 ~0.0059).
- Best BEELINE GSD F1: TRRUST prior (F1 ~0.0145).
- GRN baselines remain near-random for HPN-DREAM/BEELINE; DoRothEA+TRRUST union yields small non-zero F1
  (best ~0.029 for GENIE3/GRNBoost2).
- Perturbation-derived causal scores yield non-zero AUPR for Dixit/Shifrut but are target-list constrained.

## Codex-assisted validation (time-series + perturbation)
- Time-series validation remains dominated by priors: learned networks peak at low HPN/BEELINE F1
  (best HPN F1 ~0.0059; best BEELINE F1 ~0.00050), while OmniPath/TRRUST priors lead overall.
- AUPR fields in `summary_timeseries_metrics.csv` are now filled for priors and random baselines; priors
  use uniform scores (edge-only) so treat their AUPR as a coarse ranking-free comparison.
- Perturbation evaluation is sparse: only 19/4787 perturbations show nonzero hits, with signal
  concentrated in a handful of TFs (FOS, NEAT1, TSC22D3, JUNB, ZEB2, ZFP36).
- Implication for defaults: keep percentile thresholds conservative (>=90th percentile when F1=0)
  and continue using logistic calibration per subset; the validation signal is too weak to justify
  relaxing thresholds or switching calibration families globally.

## Phase 6 evidence table (subset, thresholds, calibration, top metrics)
Thresholds from `network_inference/outputs/sweep_results_*.json`, calibration choices from
`network_inference/outputs/calibration_report_*.json`, and metrics from
`network_inference/outputs/summary_timeseries_metrics.csv`:

| dataset subset                               | threshold (percentile, value) | calibration | HPN F1/AUPR        | BEELINE F1/AUPR     |
|:---------------------------------------------|:------------------------------|:------------|:------------------|:--------------------|
| OmniPath+DoRothEA intersection (immune HPN)  | p80 (0.0034718)               | logistic    | 0.00587 / 0.00373 | 0.000502 / 0.000289 |
| OmniPath+DoRothEA union (immune HPN)         | p95 (0.0045379)               | logistic    | 0.00177 / 0.000674 | 0.0000957 / 0.0000265 |
| OmniPath relaxed (immune HPN)                | p94 (0.0044330)               | logistic    | 0.00133 / 0.000199 | 0.0001009 / 0.00000123 |
| OmniPath intercell union (immune HPN)        | p95 (0.0044924)               | logistic    | 0.00174 / 0.000720 | 0 / 0.0000214       |
| OmniPath intercell union sources (immune HPN)| p95 (0.0045585)               | logistic    | 0.00179 / 0.000773 | 0.0000921 / 0.0000257 |
| OmniPath intercell union targets (immune HPN)| p95 (0.0045308)               | logistic    | 0.00190 / 0.000758 | 0 / 0.0000261       |

## Phase 7: Identifier alignment and reference remediation
- Crosswalk-aware alias expansion now folds ID to symbol mappings from the curated symbol-map
  crosswalks (unambiguous rows) plus MyGene-resolved IDs, so UniProt/Entrez identifiers in the
  crosswalk entries normalize directly to HGNC symbols during score evaluation.
- HPN Dream missing identifiers are now zero in the full-genes score-eval run
  (`network_inference/outputs/score_eval_probe_priors_full_genes_missing_report.json`).
- Remaining missing IDs are limited to Beeline GSD (`SRY`, `NR0B1`, `UGR`), which are out-of-domain
  for the immune gene universe; no additional remaps were added to avoid lossy symbol expansion.
- Crosswalk score-eval outputs for this pass:
  `network_inference/outputs/score_eval_probe_priors_full_genes_crosswalk.json` and
  `network_inference/outputs/score_eval_probe_priors_full_genes_crosswalk_missing_ids.tsv`.

## Network inference evaluation table (all methods)
From `network_inference/outputs/summary_timeseries_metrics.csv`:

| method                                   |   edges |   hpn_precision |   hpn_recall |      hpn_f1 | hpn_aupr               |   beeline_precision |   beeline_recall |   beeline_f1 | beeline_aupr           |
|:-----------------------------------------|--------:|----------------:|-------------:|------------:|:-----------------------|--------------------:|-----------------:|-------------:|:-----------------------|
| genie3                                   |   9410 | 0 | 0 | 0 | 3.01509312895592e-06 | 0 | 0 | 0 | 1.23677335501786e-06 |
| grnboost2                                |   8060 | 0 | 0 | 0 | 3.01513374974668e-06 | 0 | 0 | 0 | 1.23676590430998e-06 |
| omnipath_dorothea_intersection_immune    |   3950 | 0.0030503304524656 | 0.0769230769230769 | 0.0058679706601467 | 0.0037315611576285 | 0.0002541942043721 | 0.0208333333333333 | 0.0005022601707684 | 0.0002889531342925 |
| omnipath_dorothea_intersection_scaled    |   3950 | 0.0030518819938962 | 0.0769230769230769 | 0.0058708414872798 | 0.0035034565876723 | 0 | 0 | 0 | 0.0003146627202313 |
| omnipath_dorothea_union_immune           |  42100 | 0.0008860365430206 | 0.237179487179487 | 0.0017654777525945 | 0.0006744426720532 | 4.78938671903063e-05 | 0.0416666666666666 | 9.56777573133686e-05 | 2.6544377997009e-05 |
| omnipath_dorothea_union_scaled           |  42100 | 0.0009332822819948 | 0.25 | 0.0018596223536143 | 0.0006771969327898 | 4.78606298458888e-05 | 0.0416666666666666 | 9.56114351276413e-05 | 2.77076226489599e-05 |
| omnipath_dorothea_union_scaled_1200x420  |  42100 | 0.0008374007082017 | 0.224358974358974 | 0.0016685736079328 | 0.000523751114584 | 7.17772035601493e-05 | 0.0625 | 0.000143389733295 | 2.77673713667118e-05 |
| omnipath_intercell_union                 |  43750 | 0.0008724200472943 | 0.243589743589744 | 0.0017386132271864 | 0.0007202008450681 | 0 | 0 | 0 | 2.14159337785715e-05 |
| omnipath_intercell_union_sources         |  43750 | 0.0008990939899024 | 0.25 | 0.0017917441940596 | 0.000772952742185 | 4.61073840975632e-05 | 0.0416666666666666 | 9.21128382268279e-05 | 2.57192374035326e-05 |
| omnipath_intercell_union_sources_relaxed |  61900 | 0.0006364749082007 | 0.25 | 0.0012697172437368 | 0.0006129663670445 | 3.26397388820889e-05 | 0.0416666666666666 | 6.52283808685159e-05 | 1.67394391934158e-05 |
| omnipath_intercell_union_targets         |  39950 | 0.0009556622991222 | 0.243589743589744 | 0.0019038553069966 | 0.0007583188302276 | 0 | 0 | 0 | 2.61019754360668e-05 |
| omnipath_prior                           |  84547 | 0.011070110701107 | 0.115384615384615 | 0.0202020202020202 | 0.0019695128509511 | 0.0006150061500615 | 0.0208333333333333 | 0.001194743130227 | 7.8018569331413e-06 |
| omnipath_relaxed_immune                  |  39950 | 0.000666683761122 | 0.204724409448819 | 0.001329039513367 | 0.000199377420258 | 5.04872014944212e-05 | 0.0416666666666666 | 0.0001008522010992 | 1.23345680980409e-06 |
| pearson                                  |  50000 | 0 | 0 | 0 | 3.01439206448139e-06 | 0 | 0 | 0 | 1.23673788280425e-06 |
| pidc_full                                |   9950 | 0 | 0 | 0 | 3.0147248264269e-06 | 0 | 0 | 0 | 1.23731115627571e-06 |
| pidc_proxy                               |  15000 | 0 | 0 | 0 | 3.01471731825889e-06 | 0 | 0 | 0 | 1.23767545939697e-06 |
| random                                   |  50000 | 0 | 0 | 0 | 3.01379619122445e-06 | 0 | 0 | 0 | 1.23668966307949e-06 |
| random_3950                              |   3950 | 0 | 0 | 0 | 3.01411240909184e-06 | 0 | 0 | 0 | 1.23723698534548e-06 |
| scenic_grnboost2                         |   3488 | 0 | 0 | 0 | 3.0148447640772e-06 | 0 | 0 | 0 | 1.23752251950874e-06 |
| scenic_pruned                            |     29 | 0 | 0 | 0 | 3.01475162610475e-06 | 0 | 0 | 0 | 1.23733017764485e-06 |
| spearman                                 |  50000 | 2.00400801603206e-05 | 0.0064102564102564 | 3.99552501198657e-05 | 3.0143920724456e-06 | 0 | 0 | 0 | 1.23673788280425e-06 |
| trrust_prior                             |   8406 | 0.0012853470437017 | 0.0064102564102564 | 0.0021413276231263 | 9.00638144370403e-06 | 0.0077120822622107 | 0.125 | 0.0145278450363196 | 0.0010296167496008 |

## Sweep analysis (top-k vs percentile)
From `network_inference/outputs/sweep_results_*.json`:

| sweep_file                                                             |   candidate_edges |   candidate_aupr |   best_percentile |   best_percentile_f1 |   best_top_k |   best_top_k_f1 |
|:-----------------------------------------------------------------------|------------------:|-----------------:|------------------:|---------------------:|-------------:|----------------:|
| sweep_results.json                                                     |            504210 |      0.000486346 |                90 |          0           |           10 |     0           |
| sweep_results_omnipath.json                                            |            162835 |      0.000517499 |                80 |          0.00122523  |          100 |     0.00119726  |
| sweep_results_omnipath_dorothea_intersection_immune_hpn.json           |             29253 |      0.0251236   |                80 |          0.0435188   |          150 |     0.0387678   |
| sweep_results_omnipath_dorothea_union_immune_hpn.json                  |            870806 |      0.00127928  |                95 |          0.00293374  |          100 |     0.00185946  |
| sweep_results_omnipath_dorothea_union_immune_hpn_scaled.json           |            870806 |      0.00126761  |                80 |          0.00276677  |           50 |     0.00181941  |
| sweep_results_omnipath_dorothea_union_immune_hpn_scaled_1200x420.json  |            870806 |      0.00122933  |                94 |          0.00252735  |           25 |     0.00219972  |
| sweep_results_omnipath_intercell_moderate_immune_hpn.json              |             64942 |      4.15643e-05 |                80 |          0           |          150 |     9.8032e-05  |
| sweep_results_omnipath_intercell_relaxed_immune_hpn.json               |            202258 |      0.000637265 |                90 |          0.00177322  |          100 |     0.000809425 |
| sweep_results_omnipath_intercell_strict_immune_hpn.json                |             42197 |      6.32627e-05 |                80 |          0           |          150 |     0.000130069 |
| sweep_results_omnipath_intercell_union_immune_hpn.json                 |            866520 |      0.00110004  |                95 |          0.00245907  |           25 |     0.00160228  |
| sweep_results_omnipath_intercell_union_sources_immune_hpn.json         |            748424 |      0.00115573  |                95 |          0.00263123  |          100 |     0.00161212  |
| sweep_results_omnipath_intercell_union_sources_relaxed_immune_hpn.json |           1059089 |      0.000939412 |                85 |          0.0020195   |           25 |     0.0011417   |
| sweep_results_omnipath_intercell_union_targets_immune_hpn.json         |            791235 |      0.00116319  |                95 |          0.00263977  |           25 |     0.00175029  |
| sweep_results_omnipath_relaxed.json                                    |            495027 |      0.000446642 |                80 |          0.000886918 |          150 |     0.000694634 |
| sweep_results_omnipath_relaxed_immune_hpn.json                         |            648849 |      0.0013050   |                94 |          0.0028351   |          100 |     0.0018348   |
| sweep_results_regulatory.json                                          |             43198 |      0.00859106  |                85 |          0.019143    |          100 |     0.0170197   |

Top-k sweeps rarely beat percentile selections; only intercell moderate/strict show tiny nonzero F1
(~1e-4) when percentile F1 is zero, which is too small to justify switching defaults.

Recommended percentile thresholds (using best-percentile selection; if F1=0, keep the most conservative percentile):

| sweep_file                                                             |   best_percentile |   best_threshold |
|:-----------------------------------------------------------------------|------------------:|-----------------:|
| sweep_results.json                                                     |                99 |       0.00948889 |
| sweep_results_omnipath.json                                            |                80 |       0.00215086 |
| sweep_results_regulatory.json                                          |                85 |       0.00213534 |
| sweep_results_omnipath_relaxed.json                                    |                80 |       0.00213412 |
| sweep_results_omnipath_dorothea_intersection_immune_hpn.json           |                80 |       0.0034718  |
| sweep_results_omnipath_dorothea_union_immune_hpn.json                  |                95 |       0.00453789 |
| sweep_results_omnipath_dorothea_union_immune_hpn_scaled.json           |                80 |       0.00327022 |
| sweep_results_omnipath_dorothea_union_immune_hpn_scaled_1200x420.json  |                94 |       0.0044734  |
| sweep_results_omnipath_relaxed_immune_hpn.json                         |                94 |       0.0044320  |
| sweep_results_omnipath_intercell_relaxed_immune_hpn.json               |                90 |       0.00371819 |
| sweep_results_omnipath_intercell_moderate_immune_hpn.json              |                95 |       0.00422569 |
| sweep_results_omnipath_intercell_strict_immune_hpn.json                |                95 |       0.00420093 |
| sweep_results_omnipath_intercell_union_immune_hpn.json                 |                95 |       0.00449238 |
| sweep_results_omnipath_intercell_union_sources_immune_hpn.json         |                95 |       0.00455853 |
| sweep_results_omnipath_intercell_union_sources_relaxed_immune_hpn.json |                85 |       0.0033032038 |
| sweep_results_omnipath_intercell_union_targets_immune_hpn.json         |                95 |       0.00453081 |

## Calibration reports (all runs)
From `network_inference/outputs/calibration_report_*.json`:
Calibration-delta scan highlights: the largest ECE drop is on the OmniPath+DoRothEA intersection (immune HPN), the highest Brier subset is flagged, and isotonic vs logistic Brier deltas remain consistently tiny despite ECE drift.

| report                                                                     | method   |   n_edges_total |   n_edges_fit |   brier_score |         ece |
|:---------------------------------------------------------------------------|:---------|----------------:|--------------:|--------------:|------------:|
| calibration_report_omnipath_dorothea_intersection_immune_hpn_isotonic.json | isotonic |           29253 |         29253 |   0.0195222   | 0.0050485   |
| calibration_report_omnipath_dorothea_intersection_immune_hpn_logistic.json | logistic |           29253 |         29253 |   0.0195324   | 1.37017e-08 |
| calibration_report_omnipath_dorothea_union_immune_hpn_isotonic.json        | isotonic |          870806 |        870806 |   0.000884422 | 0.000127866 |
| calibration_report_omnipath_dorothea_union_immune_hpn_logistic.json        | logistic |          870806 |        870806 |   0.000884603 | 5.26213e-11 |
| calibration_report_omnipath_isotonic.json                                  | isotonic |          162835 |        162835 |   0.000491041 | 4.03484e-09 |
| calibration_report_omnipath_logistic.json                                  | logistic |          162835 |        162835 |   0.000491053 | 2.27127e-13 |
| calibration_report_omnipath_intercell_union_immune_hpn_isotonic.json        | isotonic |          866520 |        866520 |   0.000683863 | 0.000320803 |
| calibration_report_omnipath_intercell_union_immune_hpn_logistic.json        | logistic |          866520 |        866520 |   0.000683878 | 3.63338e-11 |
| calibration_report_omnipath_intercell_union_sources_immune_hpn_isotonic.json | isotonic |          748424 |        748424 |   0.000778183 | 3.93559e-05 |
| calibration_report_omnipath_intercell_union_sources_immune_hpn_logistic.json | logistic |          748424 |        748424 |   0.000778363 | 4.40298e-11 |
| calibration_report_omnipath_intercell_union_targets_immune_hpn_isotonic.json | isotonic |          791235 |        791235 |   0.000749088 | 0.000477903 |
| calibration_report_omnipath_intercell_union_targets_immune_hpn_logistic.json | logistic |          791235 |        791235 |   0.0007489   | 4.27649e-11 |
| calibration_report_omnipath_relaxed_immune_hpn_isotonic.json               | isotonic |          648849 |        648849 |   0.000883662 | 4.90723e-05 |
| calibration_report_omnipath_relaxed_immune_hpn_logistic.json               | logistic |          648849 |        648849 |   0.000883861 | 4.60276e-11 |
| calibration_report_omnipath_relaxed_isotonic.json                          | isotonic |          475801 |        475801 |   0.00044745  | 1.17853e-05 |
| calibration_report_omnipath_relaxed_logistic.json                          | logistic |          475801 |        475801 |   0.000447466 | 1.73647e-12 |
| calibration_report_regulatory_isotonic.json                                | isotonic |           43198 |         43198 |   0.0071427   | 5.73154e-06 |
| calibration_report_regulatory_logistic.json                                | logistic |           43198 |         43198 |   0.00714758  | 5.99367e-10 |

## GRN baselines (detailed)
From `network_inference/outputs/score_eval_grn_baselines_immune.csv` (reference label
`dorothea_trrust_union_immune` denotes the immune-filtered DoRothEA+TRRUST union):

| method           | reference                    | precision   | recall      | f1          | aupr        | pred_edges | true_edges | candidate_edges |
|:-----------------|:-----------------------------|:------------|:------------|:------------|:------------|-----------:|-----------:|----------------:|
| genie3           | hpn_dream                    | 0           | 0           | 0           | 3.01509e-06 |       9410 |        127 |        24408540 |
| genie3           | beeline_gsd                  | 0           | 0           | 0           | 1.23677e-06 |       9410 |         48 |        24408540 |
| genie3           | dorothea_trrust_union_immune | 0.0323061   | 0.0262454   | 0.028962    | 0.00130422  |       9410 |      11583 |        24408540 |
| grnboost2        | hpn_dream                    | 0           | 0           | 0           | 3.01513e-06 |       8060 |        127 |        24408540 |
| grnboost2        | beeline_gsd                  | 0           | 0           | 0           | 1.23677e-06 |       8060 |         48 |        24408540 |
| grnboost2        | dorothea_trrust_union_immune | 0.0348635   | 0.0242597   | 0.0286107   | 0.00162393  |       8060 |      11583 |        24408540 |
| scenic_grnboost2 | hpn_dream                    | 0           | 0           | 0           | 3.01484e-06 |       3488 |        127 |        24408540 |
| scenic_grnboost2 | beeline_gsd                  | 0           | 0           | 0           | 1.23752e-06 |       3488 |         48 |        24408540 |
| scenic_grnboost2 | dorothea_trrust_union_immune | 0.0229358   | 0.00690667  | 0.0106164   | 0.000454317 |       3488 |      11583 |        24408540 |
| scenic_pruned    | hpn_dream                    | 0           | 0           | 0           | 3.01475e-06 |         29 |        127 |        24408540 |
| scenic_pruned    | beeline_gsd                  | 0           | 0           | 0           | 1.23733e-06 |         29 |         48 |        24408540 |
| scenic_pruned    | dorothea_trrust_union_immune | 0.047619    | 8.63334e-05 | 0.000172354 | 0.000319505 |         29 |      11583 |        24408540 |
| pidc_proxy       | hpn_dream                    | 0           | 0           | 0           | 3.01472e-06 |      15000 |        127 |        24408540 |
| pidc_proxy       | beeline_gsd                  | 0           | 0           | 0           | 1.23768e-06 |      15000 |         48 |        24408540 |
| pidc_proxy       | dorothea_trrust_union_immune | 0.0022      | 0.002849    | 0.00248279  | 0.000361978 |      15000 |      11583 |        24408540 |
| pidc_full        | hpn_dream                    | 0           | 0           | 0           | 3.01472e-06 |       9950 |        127 |        24408540 |
| pidc_full        | beeline_gsd                  | 0           | 0           | 0           | 1.23731e-06 |       9950 |         48 |        24408540 |
| pidc_full        | dorothea_trrust_union_immune | 0           | 0           | 0           | 0.000310864 |       9950 |      11583 |        24408540 |
| pearson          | hpn_dream                    | 0           | 0           | 0           | 3.01439e-06 |      49900 |        127 |        24408540 |
| pearson          | beeline_gsd                  | 0           | 0           | 0           | 1.23674e-06 |      49900 |         48 |        24408540 |
| pearson          | dorothea_trrust_union_immune | 0.00132265  | 0.00569801  | 0.00214693  | 0.000342043 |      49900 |      11583 |        24408540 |
| spearman         | hpn_dream                    | 0           | 0           | 0           | 3.01439e-06 |      49900 |        127 |        24408540 |
| spearman         | beeline_gsd                  | 0           | 0           | 0           | 1.23674e-06 |      49900 |         48 |        24408540 |
| spearman         | dorothea_trrust_union_immune | 0.00134269  | 0.00578434  | 0.00217946  | 0.000343395 |      49900 |      11583 |        24408540 |
| random           | hpn_dream                    | 0           | 0           | 0           | 3.0138e-06  |      49809 |        127 |        24408540 |
| random           | beeline_gsd                  | 0           | 0           | 0           | 1.23669e-06 |      49809 |         48 |        24408540 |
| random           | dorothea_trrust_union_immune | 0.000481841 | 0.002072    | 0.000781861 | 0.000311952 |      49809 |      11583 |        24408540 |

## Perturbation validation (detailed)
From `single_cell_mechinterp/outputs/perturb_validation/*/perturbation_metrics.tsv`:

| run                                     | intervention   |   n_pairs |   n_pos |       aupr |      auroc |   perm_p_value | score_mode   |
|:----------------------------------------|:---------------|----------:|--------:|-----------:|-----------:|---------------:|:-------------|
| adamson_causal_adamson                  | ablation       |        31 |      18 |   0.576087 |   0.482906 |       0.40796  | abs          |
| adamson_causal_adamson                  | swap           |        31 |      18 |   0.576906 |   0.521368 |       0.378109 | abs          |
| adamson_causal_immune                   | ablation       |         0 |       0 |   nan      |   nan      |       nan      | abs          |
| adamson_causal_immune                   | swap           |         0 |       0 |   nan      |   nan      |       nan      | abs          |
| dixit7_causal_dixit_targets_large       | ablation       |       400 |     219 |   0.541526 |   0.488181 |       0.621891 | abs          |
| dixit7_causal_dixit_targets_large       | swap           |       400 |     219 |   0.538596 |   0.486642 |       0.666667 | abs          |
| dixit7_causal_dixit_targets_large_fast  | ablation       |       400 |     219 |   0.491849 |   0.5      |       0.99005  | abs          |
| dixit7_causal_dixit_targets_tiny_neg    | ablation       |        40 |      28 |   0.771593 |   0.589286 |       0.114428 | abs          |
| dixit7_causal_dixit_targets_tiny_neg    | swap           |        40 |      28 |   0.7047   |   0.568452 |       0.368159 | abs          |
| dixit_causal_adamson                    | ablation       |         0 |       0 |   nan      |   nan      |       nan      | abs          |
| dixit_causal_adamson                    | swap           |         0 |       0 |   nan      |   nan      |       nan      | abs          |
| dixit_causal_dixit_targets_large        | ablation       |       400 |     318 |   0.812835 |   0.53904  |       0.208955 | abs          |
| dixit_causal_dixit_targets_large        | swap           |       400 |     318 |   0.747166 |   0.445448 |       0.995025 | abs          |
| dixit_causal_dixit_targets_large_fast   | ablation       |       400 |     318 |   0.66243  |   0.5      |       1        | abs          |
| dixit_causal_dixit_targets_tiny_neg     | ablation       |        40 |      33 |   0.838517 |   0.549784 |       0.273632 | abs          |
| dixit_causal_dixit_targets_tiny_neg     | swap           |        40 |      33 |   0.737049 |   0.350649 |       0.925373 | abs          |
| shifrut_causal_shifrut_targets_tiny_neg | ablation       |        40 |      36 |   0.794021 |   0.40625  |       0.9801   | abs          |
| shifrut_causal_shifrut_targets_tiny_neg | swap           |        40 |      36 |   0.876374 |   0.576389 |       0.58209  | abs          |

Notes:
- Runs with n_pairs=0 have no overlap after mapping and are reported as NaN.
- Dixit fast run has all-zero effect_mean values; interpret as a sanity check only.

## Probe benchmark (mechanistic interpretability)
AUPR by probe and reference from `single_cell_mechinterp/outputs/probe_benchmark/probe_metrics.csv`:

| probe                |   dorothea |   dorothea_chipseq |     trrust |
|:---------------------|-----------:|-------------------:|-----------:|
| attention            |  0.0478443 |          0.0270738 | 0.00178467 |
| consensus            |  0.0487764 |          0.0275119 | 0.00176285 |
| grad_input           |  0.0415129 |          0.0197037 | 0.00166252 |
| integrated_gradients |  0.0333836 |          0.0218558 | 0.00149142 |
| perturbation         |  0.0354776 |          0.0216272 | 0.00230225 |

Score-eval summary for probe priors and attention-inferred edges
(`network_inference/outputs/score_eval_probe_priors.json`):
- DoRothEA overlap is small but nonzero across probe priors; best F1 is `probe_consensus`
  (~3.00e-4, precision ~1.54e-4, recall ~4.91e-3).
- TRRUST overlap remains near-zero; only `probe_integrated_gradients` yields a nonzero F1
  (~8.34e-6) with AUPR ~1.74e-5, while other methods are effectively zero.
Gated-overlap summary for the full immune gene universe
(`network_inference/outputs/score_eval_probe_priors_full_genes.json`):
- HPN Dream remains at zero precision/recall/F1 across all methods (true_edges=232); AUPR is
  gated off because candidate edges exceed 2.5B.
- DoRothEA retains small but nonzero overlap across probes; best F1 is `probe_consensus`
  (1.48e-4, precision 1.64e-4, recall 1.35e-4).
- TRRUST remains near-zero; only `probe_integrated_gradients` has nonzero F1 (8.64e-6,
  precision 4.48e-6, recall 1.20e-4).
Overlap-gate exclusions (requires >=1% ref-node overlap and >=0.1% gene-universe overlap):
- Base gene universe (`score_eval_probe_priors.json`): HPN Dream excluded with 3/51 nodes
  overlapping (5.88%) and 0.0625% gene-universe overlap; Beeline GSD excluded with 2/17 nodes
  (11.76%) and 0.0417% gene-universe overlap.
- Full-genes crosswalk (`score_eval_probe_priors_full_genes_crosswalk.json`): Beeline GSD
  excluded with 15/17 nodes (88.24%) but only 0.0301% gene-universe overlap.
- No overlap-gate exclusions in `score_eval_probe_priors_full_genes.json` (HPN Dream
  gene-universe overlap 0.102%) or `score_eval_grn_baselines_immune.json` (HPN Dream 0.770%,
  Beeline GSD 0.304%).

Coexpression sanity from `single_cell_mechinterp/outputs/probe_benchmark/coexpression_sanity.csv`:

| probe                |   n_edges |   mean_corr |   median_corr |   frac_pos |   frac_gt_0_2 |   baseline_mean_corr |   baseline_median_corr |   baseline_frac_pos |   baseline_frac_gt_0_2 |
|:---------------------|----------:|------------:|--------------:|-----------:|--------------:|---------------------:|-----------------------:|--------------------:|-----------------------:|
| attention            |      5250 | -0.00470724 |  -0.00198932  |   0.448762 |      0.172952 |            0.0232217 |            0           |            0.455429 |               0.159619 |
| grad_input           |      5250 |  0.0144494  |  -0.00178133  |   0.42381  |      0.180571 |            0.0231627 |            0           |            0.458857 |               0.172381 |
| integrated_gradients |      5250 |  0.015586   |  -0.0015096   |   0.426857 |      0.183238 |            0.0188069 |           -0.000500249 |            0.437524 |               0.160762 |
| perturbation         |      5250 |  0.0163942  |  -0.00150955  |   0.427238 |      0.183429 |            0.0187621 |            0           |            0.440762 |               0.161524 |
| consensus            |      5250 |  0.00312921 |  -0.000958211 |   0.458667 |      0.184    |            0.0217729 |            0           |            0.460952 |               0.168571 |

Additional probe benchmark outputs:
- `single_cell_mechinterp/outputs/probe_benchmark/probe_threshold_sweep.csv`
- `single_cell_mechinterp/outputs/probe_benchmark/probe_agreement.tsv`
- `single_cell_mechinterp/outputs/probe_benchmark/consensus_analysis.csv`
- `single_cell_mechinterp/outputs/probe_benchmark/consensus_ablation_summary.csv`

## Causal intervention summary (MI)
From `single_cell_mechinterp/outputs/causal_metrics_summary.tsv`:

| group              | reference      | score_source   | intervention   |   n_runs |   aupr_mean |   aupr_ci_low |   aupr_ci_high |   auroc_mean |   auroc_ci_low |   auroc_ci_high |   perm_p_median |
|:-------------------|:---------------|:---------------|:---------------|---------:|------------:|--------------:|---------------:|-------------:|---------------:|----------------:|----------------:|
| immune_v2          | dorothea       | causal         | ablation       |        2 |    0.52674  |      0.483628 |       0.569852 |     0.469091 |       0.351515 |        0.586667 |     0.672328    |
| immune_v2          | dorothea       | causal         | swap           |        2 |    0.540627 |      0.481866 |       0.599388 |     0.477879 |       0.342424 |        0.613333 |     0.620879    |
| immune_v2          | trrust_human   | causal         | ablation       |        2 |    0.59548  |      0.550418 |       0.640542 |     0.565051 |       0.517857 |        0.612245 |     0.358641    |
| immune_v2          | trrust_human   | causal         | swap           |        2 |    0.622149 |      0.53872  |       0.705578 |     0.593112 |       0.458333 |        0.727891 |     0.313187    |
| kidney_dorothea_v3 | dorothea       | causal         | ablation       |        1 |    0.782043 |    nan        |     nan        |     0.588407 |     nan        |      nan        |     0.014985    |
| kidney_dorothea_v3 | dorothea       | causal         | swap           |        1 |    0.666773 |    nan        |     nan        |     0.51815  |     nan        |      nan        |     0.611389    |
| kidney_dorothea_v3 | dorothea_human | causal         | ablation       |        1 |    0.782043 |    nan        |     nan        |     0.588407 |     nan        |      nan        |     0.022977    |
| kidney_dorothea_v3 | dorothea_human | causal         | swap           |        1 |    0.666773 |    nan        |     nan        |     0.51815  |     nan        |      nan        |     0.63037     |
| kidney_rg_v2       | dorothea       | causal         | ablation       |        1 |    0.4753   |    nan        |     nan        |     0.515497 |     nan        |      nan        |     0.584416    |
| kidney_rg_v2       | trrust_human   | causal         | ablation       |        1 |    0.540028 |    nan        |     nan        |     0.559942 |     nan        |      nan        |     0.263736    |
| kidney_v2          | dorothea       | causal         | ablation       |        4 |    0.536855 |      0.50525  |       0.562953 |     0.57386  |       0.542185 |        0.599282 |     0.18032     |
| kidney_v2          | dorothea       | causal         | swap           |        4 |    0.508522 |      0.467659 |       0.549386 |     0.500763 |       0.467235 |        0.536832 |     0.511489    |
| kidney_v2          | trrust_human   | causal         | ablation       |        4 |    0.601842 |      0.575321 |       0.628363 |     0.571713 |       0.560721 |        0.590951 |     0.105395    |
| kidney_v2          | trrust_human   | causal         | swap           |        4 |    0.536648 |      0.512852 |       0.562957 |     0.483449 |       0.464162 |        0.502736 |     0.554945    |
| krasnow_lung_v4    | dorothea       | causal         | ablation       |        1 |    0.625466 |    nan        |     nan        |     0.353846 |     nan        |      nan        |     0.769231    |
| krasnow_lung_v4    | dorothea       | causal         | swap           |        1 |    0.740251 |    nan        |     nan        |     0.630769 |     nan        |      nan        |     0.242757    |
| krasnow_lung_v4    | trrust_human   | causal         | ablation       |        1 |    0.363476 |    nan        |     nan        |     0.246914 |     nan        |      nan        |     0.934066    |
| krasnow_lung_v4    | trrust_human   | causal         | swap           |        1 |    0.497666 |    nan        |     nan        |     0.555556 |     nan        |      nan        |     0.371628    |
| lung_hi_v1         | dorothea       | causal         | ablation       |        1 |    0.690992 |    nan        |     nan        |     0.625623 |     nan        |      nan        |     0.101898    |
| lung_hi_v1         | dorothea       | causal         | swap           |        1 |    0.576402 |    nan        |     nan        |     0.462612 |     nan        |      nan        |     0.893107    |
| lung_hi_v1         | trrust_human   | causal         | ablation       |        1 |    0.757411 |    nan        |     nan        |     0.630542 |     nan        |      nan        |     0.003996    |
| lung_hi_v1         | trrust_human   | causal         | swap           |        1 |    0.666789 |    nan        |     nan        |     0.547291 |     nan        |      nan        |     0.172827    |
| lung_rg_v2         | dorothea       | causal         | ablation       |        1 |    0.682616 |    nan        |     nan        |     0.580601 |     nan        |      nan        |     0.114885    |
| lung_rg_v2         | trrust_human   | causal         | ablation       |        1 |    0.673495 |    nan        |     nan        |     0.610206 |     nan        |      nan        |     0.00699301  |
| lung_seed7_fast_v1 | dorothea       | causal         | ablation       |        1 |    0.691466 |    nan        |     nan        |     0.628788 |     nan        |      nan        |     0.208791    |
| lung_seed7_fast_v1 | dorothea       | causal         | swap           |        1 |    0.646007 |    nan        |     nan        |     0.526515 |     nan        |      nan        |     0.402597    |
| lung_seed7_fast_v1 | trrust_human   | causal         | ablation       |        1 |    0.654343 |    nan        |     nan        |     0.545455 |     nan        |      nan        |     0.368631    |
| lung_seed7_fast_v1 | trrust_human   | causal         | swap           |        1 |    0.658445 |    nan        |     nan        |     0.484848 |     nan        |      nan        |     0.345654    |
| lung_v2            | dorothea       | causal         | ablation       |        1 |    0.781984 |    nan        |     nan        |     0.687969 |     nan        |      nan        |     0.000999001 |
| lung_v2            | dorothea       | causal         | swap           |        1 |    0.614058 |    nan        |     nan        |     0.521544 |     nan        |      nan        |     0.460539    |
| lung_v2            | trrust_human   | causal         | ablation       |        1 |    0.727227 |    nan        |     nan        |     0.713657 |     nan        |      nan        |     0.000999001 |
| lung_v2            | trrust_human   | causal         | swap           |        1 |    0.552405 |    nan        |     nan        |     0.516551 |     nan        |      nan        |     0.423576    |

## Head/layer evidence summary
Head/layer evidence outputs:
- `network_inference/outputs/head_layer_evidence.tsv` (240,150 edges)
- `network_inference/outputs/head_layer_evidence_top_10k.tsv` (10,000 edges)
- `network_inference/outputs/head_layer_evidence_top_2k.tsv` (2,000 edges)

Atlas comparison note: atlas head/layer means computed over all entries were dominated by zero mass in
immune/external datasets, producing negative or near-zero correlations against evidence means. When the
atlas mean is computed over nonzero entries only, Pearson correlations become consistently positive across
immune/kidney/lung/external, while Spearman remains weak or mixed. This suggests the evidence TSVs align
with relative magnitude among active (nonzero) attention values, but rank-order agreement across all heads
is limited once zeros are included.
Counts-based nonzero-only atlas means show weaker, mixed agreement: Pearson flips negative for immune/kidney
and only turns modestly positive for lung/external, with Spearman staying low except for external. This
contrasts with the score-based nonzero-only correlations, where Pearson stays positive across all datasets.
Atlas correlation summary (`network_inference/outputs/head_layer_atlas_correlations.csv`): mean-all
correlations remain negative for immune/lung/external (kidney is modestly positive), while mean-nonzero
correlations shift to consistently positive Pearson (roughly 0.45 to 0.70). Spearman remains weak in both
cases, and atlas zero fractions are high in immune/external (0.96/0.92), indicating zero mass drives the
sign flips in mean-all comparisons.
External Krasnow lung swap delta note: the `tf_source_only`-only heads outperform their swapped
`unconstrained` counterparts by +0.0074 to +0.0096 max F1, while the `unconstrained`-only heads are ~-0.0036
lower than their `tf_source_only` F1, so the swaps are driven by moderate-to-large gains on the
`tf_source_only` side.

| dataset               | evidence_set | mean_type    |   pearson |  spearman |
|:----------------------|:-------------|:-------------|----------:|----------:|
| immune                | top_2k       | mean_all     | -0.378544 | -0.371460 |
| immune                | top_2k       | mean_nonzero |  0.695787 | -0.264840 |
| kidney                | top_2k       | mean_all     |  0.207789 |  0.102523 |
| kidney                | top_2k       | mean_nonzero |  0.689279 |  0.106660 |
| lung                  | top_2k       | mean_all     | -0.330825 |  0.113199 |
| lung                  | top_2k       | mean_nonzero |  0.672922 |  0.231470 |
| external_krasnow_lung | top_2k       | mean_all     | -0.246606 |  0.075326 |
| external_krasnow_lung | top_2k       | mean_nonzero |  0.699195 |  0.196460 |

## Figures
- `network_inference/outputs/figures/hpn_beeline_f1_scatter.png` — HPN-DREAM vs BEELINE GSD F1 scatter across all methods.
- `network_inference/outputs/figures/hpn_f1_top10_bar.png` — Top 10 methods by HPN-DREAM F1.
- `network_inference/outputs/figures/beeline_f1_top10_bar.png` — Top 10 methods by BEELINE GSD F1.
- `network_inference/outputs/figures/sweep_best_f1_bar.png` — Best F1 per sweep: percentile vs top-k.
- `network_inference/outputs/figures/sweep_percentile_curves.png` — Percentile sweep F1 curves for key sweeps.
- `network_inference/outputs/figures/sweep_candidate_edges_aupr.png` — Candidate edges vs candidate AUPR for sweeps.
- `network_inference/outputs/figures/calibration_brier_ece_scatter.png` — Calibration summary: Brier score vs ECE (log scale).
- `network_inference/outputs/figures/perturbation_aupr_auroc.png` — Perturbation validation AUPR and AUROC by run and intervention.
- `network_inference/outputs/figures/probe_aupr_heatmap.png` — Probe benchmark AUPR heatmap across references.
- `network_inference/outputs/figures/coexpression_mean_corr.png` — Coexpression sanity: mean correlation vs baseline.
- `network_inference/outputs/figures/causal_aupr_mean.png` — Causal metrics summary: AUPR mean by group/reference/intervention.
- `network_inference/outputs/figures/head_layer_score_distributions.png` — Head/layer evidence score distributions (top 10k edges).

## Limitations
- Curated priors (OmniPath/TRRUST) remain stronger baselines than attention-derived networks on F1.
- OmniPath missing IDs are dominated by underscore-joined complexes or fusion-like identifiers; the policy is to avoid splitting them by default, with an optional `omnipath.exclude_underscore_composites` gate to drop those edges when building candidate masks or proxy edges.
- Beeline GSD is out-of-domain for the immune gene universe (missing `SRY`, `NR0B1`, `UGR`), so full-genes score evaluation excludes it and avoids lossy symbol remaps.
- Node overlap for HPN-DREAM/Beeline against immune inferred/probe node sets remains ~0-2 genes, so overlap-driven F1/AUPR are effectively out-of-domain without new mapping or a matched reference.
- Candidate gating biases evaluation and inflates recall.
- Perturbation validation for Dixit/Shifrut uses perturbation-derived target lists and forced coverage.
- Fast Dixit run has degenerate causal scores (all zero).
- Larger-scale attention extraction (>=600 cells, >1200 genes) did not complete on CPU; GPU is required.

## Key artifacts (expanded)
- Network evaluation: `network_inference/outputs/summary_timeseries_metrics.csv`
- Sweep outputs: `network_inference/outputs/sweep_results_*.json`
- Calibration reports: `network_inference/outputs/calibration_report_*.json`
- Baseline evaluations: `network_inference/outputs/baseline_eval_hpn_beeline.json`,
  `network_inference/outputs/grn_baseline_eval_hpn_beeline.json`
- GRN baseline scores: `network_inference/outputs/score_eval_grn_baselines_immune.csv`
- Inferred edges: `network_inference/outputs/inferred_edges*.tsv`, `network_inference/outputs/inferred_edges*.graphml`
- Run manifests: `network_inference/outputs/run_manifest_*.json`
- Perturbation validation: `single_cell_mechinterp/outputs/perturb_validation/*/perturbation_metrics.tsv`
- Probe benchmark: `single_cell_mechinterp/outputs/probe_benchmark/*`
- Causal metrics summary: `single_cell_mechinterp/outputs/causal_metrics_summary.tsv`

## Reproducibility (expanded commands)
```
# Attention extraction (immune subset)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.cli --config configs/immune_subset_hpn.yaml prepare-data
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.cli --config configs/immune_subset_hpn.yaml extract-attention --batch-size 2 --device cpu

# Network inference and evaluation
python -m network_inference.src.cli infer --config network_inference/configs/omnipath_signaling_dorothea_intersection_immune_hpn.yaml
python -m network_inference.src.cli evaluate-timeseries --config network_inference/configs/timeseries_eval_hpn_dream_dorothea_intersection_immune.yaml
python -m network_inference.src.cli evaluate-timeseries --config network_inference/configs/timeseries_eval_beeline_gsd_dorothea_intersection_immune.yaml

# Sweep and calibration
python -m network_inference.src.cli sweep --config network_inference/configs/omnipath_sweep_dorothea_union_immune_hpn_scaled_1200x420.yaml
python -m network_inference.src.cli calibrate --config network_inference/configs/calibration_omnipath_dorothea_union_immune_hpn_logistic.yaml

# GRN baseline scoring
python -m network_inference.src.cli evaluate-scores --config network_inference/configs/score_eval_grn_baselines_immune.yaml

# Report table validation
python network_inference/scripts/validate_report_tables.py --pdf network_inference/REPORT.pdf --timeseries network_inference/outputs/summary_timeseries_metrics.csv --grn network_inference/outputs/score_eval_grn_baselines_immune.csv --log network_inference/outputs/report_table_validation.log

# Dixit causal interventions (full 200+200) + validation
PYTHONPATH=. python scripts/run_causal_interventions.py --config configs/causal_intervention_dixit_targets_large.yaml --device cpu
PYTHONPATH=. python scripts/evaluate_perturbation_validation.py --config configs/perturbation_validation_dixit_causal_dixit_targets_large.yaml --permutations 200
PYTHONPATH=. python scripts/evaluate_perturbation_validation.py --config configs/perturbation_validation_dixit7_causal_dixit_targets_large.yaml --permutations 200
```

## Next steps
- Run GPU attention extraction at larger cell and gene budgets to test scaling effects.
- Add out-of-sample perturbation benchmarks with non-overlapping target sets.
- Expand perturbation validation to additional tissues or scGPT checkpoints.
