# Data Provenance — BCB2026 Revision

Pinned for reproducibility. Every numeric claim in the paper must be traceable to one of these files.

## HGNC symbol-mapping dictionary

- Path: `single_cell_mechinterp/external/hgnc_complete_set.txt`
- Schema: 3 columns — `Approved symbol`, `Previous symbols`, `Alias symbols` (TSV)
- Rows: 50,094 (including header)
- mtime (file timestamp on disk): 2026-01-11 02:06:29 UTC
- SHA-256: `b9482a5247a4162017bd9731ac6300757b9b7511ec9a02f99bafd6b4a43a89fc`
- Source: HGNC monthly archive snapshot. The current 3-column file is a derived subset; the official monthly TSV is at `https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/archive/monthly/tsv/`. For citation, use the HGNC release matching the mtime month (2026-01).

Mapping policies (declared in §3.2 of paper):
1. `lexicographic` — ambiguous aliases resolved to the lexicographically first approved symbol
2. `drop_ambiguous` — ambiguous aliases dropped from edge set
3. `drop_unmapped` — symbols not resolvable to any approved symbol are dropped

## BEELINE reference networks (Zenodo record 3701939)

- Path: `subproject_02_evaluation_bias_protocol/implementation/data/beeline/Networks/`
- Source: https://zenodo.org/record/3701939
- Archive: `BEELINE-Networks.zip` (16 MB), SHA-256: `d54fc6ed6529141e85f94803d2c730a8173a3e867026227528ff1ae832765607`
- Files used:
  - `human-tfs.csv` — 1,563 human transcription factors (TF list T)
  - `Networks/human/hESC-ChIP-seq-network.csv` — 436,563 edges (hESC cell-type-specific ChIP-seq)
  - `Networks/human/HepG2-ChIP-seq-network.csv` — 342,862 edges (HepG2 = hHep cell-type-specific ChIP-seq)
  - `Networks/human/Non-specific-ChIP-seq-network.csv` — non-specific ChIP-seq gold standard
  - `Networks/human/STRING-network.csv` — STRING-based gold standard

## BEELINE scRNA-seq inputs (Zenodo record 3701939)

- Path: `subproject_02_evaluation_bias_protocol/implementation/data/beeline/`
- Archive: `BEELINE-data.zip` (250 MB on disk), SHA-256: `1fc849499492c25ac72514e79a339c39aa057e58cd272097450875cecb380afa`
- Datasets used:
  - **hESC** (Chu 2016, GSE75748) — `BEELINE-data/inputs/scRNA-Seq/hESC/`: ExpressionData.csv (17,735 genes × 758 cells, log-normalized), GeneOrdering.csv (17,735 rows), PseudoTime.csv (758 rows × 5 trajectory columns).
  - **hHep** (Camp 2017, GSE81252) — `BEELINE-data/inputs/scRNA-Seq/hHep/`: ExpressionData.csv (11,515 genes × 425 cells, log-normalized), GeneOrdering.csv (11,515 rows), PseudoTime.csv (425 rows × 4 columns).
- BEELINE inputs are already log-normalized (the file `ExpressionData.csv` contains log values, not raw counts). For scGPT/Geneformer extraction we may need to invert the log-normalization or use the published BEELINE preprocessing pipeline to obtain raw counts — to be decided in §3.3 of the paper.

## scGPT checkpoint

- Path: `single_cell_mechinterp/external/scGPT_checkpoints/whole-human/best_model.pt`
- Size: 196 MB
- Provenance: scGPT whole-human checkpoint published with Cui et al. 2024 (Nature Methods)

## Geneformer checkpoint

- To be downloaded — record path, size, SHA, and HuggingFace commit hash here.

## ENCODE ChIP-seq peaks (for §B6 overlap check)

- To be downloaded per TF / cell-line — record ENCODE accession numbers per peak file here.
