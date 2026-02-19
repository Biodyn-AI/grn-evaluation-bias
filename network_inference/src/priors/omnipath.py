from __future__ import annotations

from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve

import pandas as pd

from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def _resolve_column(columns: Dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "t", "yes", "y"})


def load_omnipath(
    path: str | Path,
    alias_map: Dict[str, str] | None = None,
    directed_only: bool = True,
    exclude_underscore_composites: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    columns = {col.lower(): col for col in df.columns}
    source_col = _resolve_column(columns, ("source_genesymbol", "genesymbol_a", "source", "from"))
    target_col = _resolve_column(columns, ("target_genesymbol", "genesymbol_b", "target", "to"))

    if source_col is None or target_col is None:
        raise ValueError("OmniPath file must include source/target columns")

    if directed_only and "is_directed" in columns:
        df = df[_bool_series(df[columns["is_directed"]])]

    edges = df[[source_col, target_col]].dropna().copy()
    edges.columns = ["source", "target"]
    if exclude_underscore_composites:
        source_has = edges["source"].astype(str).str.contains("_")
        target_has = edges["target"].astype(str).str.contains("_")
        mask = ~source_has & ~target_has
        edges = edges[mask]

    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_edges

    edges = normalize_edges(edges, alias_map or {})
    return edges.drop_duplicates()


def download_omnipath(
    path: str | Path,
    url: str = "https://omnipathdb.org/interactions?format=tsv&genesymbols=1",
    overwrite: bool = False,
) -> Path:
    out_path = Path(path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, out_path)
    return out_path


def download_omnipath_intercell(
    path: str | Path,
    url: str = "https://omnipathdb.org/intercell?format=tsv",
    overwrite: bool = False,
) -> Path:
    out_path = Path(path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, out_path)
    return out_path


def load_omnipath_intercell(
    path: str | Path,
    alias_map: Dict[str, str] | None = None,
    min_consensus_score: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    columns = {col.lower(): col for col in df.columns}
    gene_col = columns.get("genesymbol")
    if gene_col is None:
        raise ValueError("OmniPath intercell file must include genesymbol column")

    if min_consensus_score is not None and "consensus_score" in columns:
        df = df[df[columns["consensus_score"]] >= float(min_consensus_score)]

    genes = df[gene_col].astype(str)
    ensure_mechinterp_path()
    from src.eval.gene_symbols import normalize_gene_names

    gene_norm = normalize_gene_names(genes.values, alias_map or {})
    df = df.copy()
    df["gene"] = gene_norm
    return df
