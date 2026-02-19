from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from urllib.request import Request, urlopen


def normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    text = str(symbol).strip()
    if not text:
        return ""
    text = text.replace(" ", "").replace("\t", "")
    text = text.strip().strip(".")
    text = text.upper()
    if text.startswith("ENSG") and "." in text:
        text = text.split(".", 1)[0]
    return text


def _split_aliases(value: str) -> List[str]:
    if not value:
        return []
    parts = value.replace("|", ",").replace(";", ",").split(",")
    return [part.strip() for part in parts if part and part.strip()]


def load_hgnc_alias_map_extended(
    path: str | Path | None,
    extra_alias_cols: Iterable[str] | None = None,
) -> Dict[str, str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, sep="\t", dtype=str)
    symbol_col = None
    for candidate in (
        "symbol",
        "app_sym",
        "hgnc_symbol",
        "approved_symbol",
        "Approved symbol",
    ):
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        return {}

    alias_cols = []
    for candidate in (
        "alias_symbol",
        "alias_symbols",
        "prev_symbol",
        "previous_symbol",
        "Alias symbols",
        "Previous symbols",
    ):
        if candidate in df.columns:
            alias_cols.append(candidate)

    extra_alias_cols = list(extra_alias_cols or [])
    for col in extra_alias_cols:
        if col in df.columns and col not in alias_cols:
            alias_cols.append(col)

    alias_map: Dict[str, str] = {}

    def add_alias(alias: str, symbol: str) -> None:
        alias_norm = normalize_symbol(alias)
        symbol_norm = normalize_symbol(symbol)
        if alias_norm and symbol_norm and alias_norm not in alias_map:
            alias_map[alias_norm] = symbol_norm

    for _, row in df.iterrows():
        symbol = row.get(symbol_col)
        if not isinstance(symbol, str) or not symbol:
            continue
        symbol_norm = normalize_symbol(symbol)
        if symbol_norm and symbol_norm not in alias_map:
            alias_map[symbol_norm] = symbol_norm
        for col in alias_cols:
            aliases = row.get(col)
            if isinstance(aliases, str) and aliases:
                for alias in _split_aliases(aliases):
                    add_alias(alias, symbol)

    return alias_map


def extend_alias_map_from_adata(
    alias_map: Dict[str, str],
    adata,
    alias_cols: Iterable[str] | None = None,
) -> Dict[str, str]:
    alias_cols = [col for col in (alias_cols or []) if col in adata.var.columns]
    if not alias_cols:
        return alias_map

    for col in alias_cols:
        series = adata.var[col]
        if series is None:
            continue
        for symbol, alias in zip(adata.var_names, series.astype(str).values):
            alias_norm = normalize_symbol(alias)
            symbol_norm = normalize_symbol(symbol)
            if alias_norm and symbol_norm and alias_norm not in alias_map:
                alias_map[alias_norm] = symbol_norm

    return alias_map


def strip_phospho_suffix(symbol: str) -> str:
    if not symbol:
        return symbol
    # Strip phospho-style suffixes like _pY1173, _pS217_S221, _PS102.
    import re

    return re.sub(r"_p[STY][0-9A-Z_]*$", "", symbol, flags=re.IGNORECASE)


def _coerce_symbol_map_paths(value: object | None) -> List[str | Path]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item]
    return [value]  # type: ignore[list-item]


def collect_symbol_map_paths(mapping_cfg: Dict[str, object] | None) -> List[str | Path]:
    if not mapping_cfg:
        return []
    paths: List[str | Path] = []
    paths.extend(_coerce_symbol_map_paths(mapping_cfg.get("symbol_map_tsv")))
    paths.extend(_coerce_symbol_map_paths(mapping_cfg.get("symbol_map_tsvs")))
    return [path for path in paths if path]


def load_symbol_map(
    path: str | Path | Iterable[str | Path] | None,
    delimiter: str = "|",
) -> Dict[str, List[str]]:
    paths = _coerce_symbol_map_paths(path)
    if not paths:
        return {}

    symbol_map: Dict[str, List[str]] = {}
    for item in paths:
        path_obj = Path(item)
        if not path_obj.exists():
            continue
        df = pd.read_csv(path_obj, sep="\t", dtype=str)
        if "raw" not in df.columns or "mapped" not in df.columns:
            raise ValueError("symbol_map_tsv must include raw and mapped columns")
        for _, row in df.iterrows():
            raw = normalize_symbol(row.get("raw"))
            mapped_value = str(row.get("mapped") or "")
            if not raw or not mapped_value:
                continue
            mapped = [normalize_symbol(value) for value in mapped_value.split(delimiter)]
            mapped = [value for value in mapped if value]
            if not mapped:
                continue
            if raw not in symbol_map:
                symbol_map[raw] = []
            for value in mapped:
                if value not in symbol_map[raw]:
                    symbol_map[raw].append(value)
    return symbol_map


def map_symbol(
    symbol: str | None,
    mapping_cfg: Dict[str, object] | None,
    alias_map: Dict[str, str],
    symbol_map: Dict[str, List[str]],
) -> List[str]:
    if symbol is None:
        return []
    raw = str(symbol)
    cfg = mapping_cfg or {}
    if cfg.get("strip_phospho_suffix"):
        raw = strip_phospho_suffix(raw)
    raw_norm = normalize_symbol(raw)
    if not raw_norm:
        return []

    mapped = symbol_map.get(raw_norm, [raw_norm])
    mapped = [alias_map.get(value, value) for value in mapped]

    # Drop empty entries and deduplicate while preserving order.
    seen = set()
    result: List[str] = []
    for value in mapped:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)

    if cfg.get("drop_unmapped"):
        return [value for value in result if value in alias_map or value in alias_map.values()]
    return result


def map_edge_table(
    edges: pd.DataFrame,
    mapping_cfg: Dict[str, object] | None,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    if not mapping_cfg:
        return edges

    symbol_map = load_symbol_map(collect_symbol_map_paths(mapping_cfg))
    extra_cols = [col for col in edges.columns if col not in ("source", "target")]
    mapped_rows = []
    for _, row in edges.iterrows():
        sources = map_symbol(row.get("source"), mapping_cfg, alias_map, symbol_map)
        targets = map_symbol(row.get("target"), mapping_cfg, alias_map, symbol_map)
        if not sources or not targets:
            continue
        payload = {col: row[col] for col in extra_cols}
        for source in sources:
            for target in targets:
                mapped_rows.append({"source": source, "target": target, **payload})

    if not mapped_rows:
        return edges.iloc[0:0].copy()
    return pd.DataFrame(mapped_rows)


def _looks_like_uniprot(value: str) -> bool:
    if not value:
        return False
    if len(value) not in (6, 10):
        return False
    if not value.isalnum():
        return False
    has_alpha = any(ch.isalpha() for ch in value)
    has_digit = any(ch.isdigit() for ch in value)
    return has_alpha and has_digit


def collect_external_ids(values: Iterable[str | None]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        norm = normalize_symbol(value)
        if not norm:
            continue
        if norm.isdigit() or norm.startswith("ENSG") or _looks_like_uniprot(norm):
            if norm not in seen:
                seen.add(norm)
                result.append(norm)
    return result


def extend_alias_map_from_symbol_map(
    alias_map: Dict[str, str],
    symbol_map_tsv: str | Path | Iterable[str | Path] | None,
    id_mapping_cfg: Dict[str, object] | None = None,
    delimiter: str = "|",
) -> Dict[str, str]:
    if not symbol_map_tsv:
        return alias_map
    symbol_map = load_symbol_map(symbol_map_tsv, delimiter=delimiter)
    if not symbol_map:
        return alias_map

    for mapped in symbol_map.values():
        if not mapped:
            continue
        symbol_candidates = [value for value in mapped if value in alias_map.values()]
        if len(symbol_candidates) != 1:
            continue
        symbol = symbol_candidates[0]
        for value in collect_external_ids(mapped):
            if value not in alias_map:
                alias_map[value] = symbol

    id_values = collect_external_ids(
        value for mapped in symbol_map.values() for value in (mapped or [])
    )
    if id_mapping_cfg:
        alias_map = extend_alias_map_from_mygene(alias_map, id_values, id_mapping_cfg)
    return alias_map


def extend_alias_map_from_mygene(
    alias_map: Dict[str, str],
    identifiers: Iterable[str],
    cfg: Dict[str, object] | None = None,
) -> Dict[str, str]:
    cfg = cfg or {}
    identifiers = [value for value in identifiers if value]
    if not identifiers:
        return alias_map

    cache_path = cfg.get("cache_path")
    cache: Dict[str, str] = {}
    if cache_path:
        cache_file = Path(str(cache_path))
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache = {}

    pending = [value for value in identifiers if value not in cache]
    if pending:
        scopes = cfg.get(
            "scopes",
            ["ensembl.gene", "entrezgene", "uniprot.Swiss-Prot", "uniprot.TrEMBL"],
        )
        payload = {
            "q": pending,
            "scopes": ",".join(scopes) if isinstance(scopes, list) else scopes,
            "fields": "symbol",
            "species": cfg.get("species", "human"),
        }
        req = Request(
            "https://mygene.info/v3/query",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=cfg.get("timeout", 20)) as resp:
                results = json.loads(resp.read().decode("utf-8"))
        except Exception:
            results = []

        for item in results:
            query = normalize_symbol(item.get("query"))
            symbol = normalize_symbol(item.get("symbol"))
            if query and symbol:
                cache[query] = symbol

        if cache_path:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    for raw, symbol in cache.items():
        if raw and symbol and raw not in alias_map:
            alias_map[raw] = symbol

    return alias_map
