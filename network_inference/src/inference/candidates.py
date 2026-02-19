from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from network_inference.src.data.expression import expression_masks
from network_inference.src.priors.omnipath import load_omnipath, load_omnipath_intercell
from network_inference.src.utils.scm_imports import ensure_mechinterp_path


@dataclass
class CandidateConfig:
    use_sources_dorothea: bool = False
    use_targets_dorothea: bool = False
    use_sources_omnipath: bool = False
    use_targets_omnipath: bool = False
    use_sources_intercell: bool = False
    use_targets_intercell: bool = False
    candidate_mask_mode: str = "union"
    omnipath_directed_only: bool = True
    omnipath_exclude_underscore_composites: bool = False


def candidate_config_from_sections(
    network_cfg: Dict[str, object],
    override_cfg: Dict[str, object],
    omnipath_cfg: Dict[str, object],
) -> CandidateConfig:
    def pick(key: str, default: bool) -> bool:
        if key in override_cfg:
            return bool(override_cfg[key])
        return bool(network_cfg.get(key, default))

    return CandidateConfig(
        use_sources_dorothea=pick("candidate_sources_from_dorothea", False),
        use_targets_dorothea=pick("candidate_targets_from_dorothea", False),
        use_sources_omnipath=pick("candidate_sources_from_omnipath", False),
        use_targets_omnipath=pick("candidate_targets_from_omnipath", False),
        use_sources_intercell=pick("candidate_sources_from_intercell", False),
        use_targets_intercell=pick("candidate_targets_from_intercell", False),
        candidate_mask_mode=str(
            override_cfg.get("candidate_mask_mode", network_cfg.get("candidate_mask_mode", "union"))
        ),
        omnipath_directed_only=bool(omnipath_cfg.get("directed_only", True)),
        omnipath_exclude_underscore_composites=bool(
            omnipath_cfg.get("exclude_underscore_composites", False)
        ),
    )


def _combine_masks(masks: Iterable[np.ndarray | None], mode: str) -> np.ndarray | None:
    valid = [mask for mask in masks if mask is not None]
    if not valid:
        return None
    if mode == "union":
        combined = valid[0].copy()
        for mask in valid[1:]:
            combined |= mask
        return combined
    if mode == "intersection":
        combined = valid[0].copy()
        for mask in valid[1:]:
            combined &= mask
        return combined
    raise ValueError(f"Unknown candidate_mask_mode: {mode}")


def _candidate_masks_from_dorothea(
    gene_names_norm: np.ndarray,
    paths: Dict[str, Path],
    confidence_levels: Iterable[str] | None,
    use_sources: bool,
    use_targets: bool,
    alias_map: Dict[str, str],
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if not (use_sources or use_targets):
        return None, None

    dorothea_path = paths.get("dorothea_tsv")
    if not dorothea_path:
        raise ValueError("dorothea_tsv path is required for candidate filtering")

    ensure_mechinterp_path()
    from src.eval.dorothea import load_dorothea
    from src.eval.gene_symbols import normalize_edges

    true_edges = load_dorothea(dorothea_path, confidence_levels=confidence_levels)
    true_edges = normalize_edges(true_edges, alias_map)

    source_mask = None
    target_mask = None
    if use_sources:
        sources = set(true_edges["source"].unique())
        source_mask = np.array([name in sources for name in gene_names_norm], dtype=bool)
    if use_targets:
        targets = set(true_edges["target"].unique())
        target_mask = np.array([name in targets for name in gene_names_norm], dtype=bool)

    return source_mask, target_mask


def _candidate_masks_from_omnipath(
    gene_names_norm: np.ndarray,
    paths: Dict[str, Path],
    use_sources: bool,
    use_targets: bool,
    alias_map: Dict[str, str],
    directed_only: bool,
    exclude_underscore_composites: bool,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if not (use_sources or use_targets):
        return None, None

    omnipath_path = paths.get("omnipath_tsv")
    if not omnipath_path:
        raise ValueError("omnipath_tsv path is required for OmniPath filtering")

    edges = load_omnipath(
        omnipath_path,
        alias_map=alias_map,
        directed_only=directed_only,
        exclude_underscore_composites=exclude_underscore_composites,
    )

    source_mask = None
    target_mask = None
    if use_sources:
        sources = set(edges["source"].unique())
        source_mask = np.array([name in sources for name in gene_names_norm], dtype=bool)
    if use_targets:
        targets = set(edges["target"].unique())
        target_mask = np.array([name in targets for name in gene_names_norm], dtype=bool)

    return source_mask, target_mask


def _candidate_masks_from_intercell(
    gene_names_norm: np.ndarray,
    paths: Dict[str, Path],
    use_sources: bool,
    use_targets: bool,
    alias_map: Dict[str, str],
    intercell_cfg: Dict[str, object],
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if not (use_sources or use_targets):
        return None, None

    intercell_path = paths.get("omnipath_intercell_tsv")
    if not intercell_path:
        raise ValueError("omnipath_intercell_tsv path is required for intercell filtering")

    min_consensus = intercell_cfg.get("min_consensus_score")
    intercell_df = load_omnipath_intercell(
        intercell_path,
        alias_map=alias_map,
        min_consensus_score=min_consensus,
    )

    source_role = str(intercell_cfg.get("source_role", "transmitter"))
    target_role = str(intercell_cfg.get("target_role", "receiver"))

    if source_role not in intercell_df.columns:
        raise ValueError(f"intercell source_role column not found: {source_role}")
    if target_role not in intercell_df.columns:
        raise ValueError(f"intercell target_role column not found: {target_role}")

    source_mask = None
    target_mask = None
    def _bool_series(series):
        if series.dtype == bool:
            return series
        return series.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})

    if use_sources:
        sources = set(intercell_df.loc[_bool_series(intercell_df[source_role]), "gene"].unique())
        source_mask = np.array([name in sources for name in gene_names_norm], dtype=bool)
    if use_targets:
        targets = set(intercell_df.loc[_bool_series(intercell_df[target_role]), "gene"].unique())
        target_mask = np.array([name in targets for name in gene_names_norm], dtype=bool)

    return source_mask, target_mask


def build_candidate_masks(
    adata,
    paths: Dict[str, Path],
    candidate_cfg: CandidateConfig,
    confidence_levels: Iterable[str] | None,
    intercell_cfg: Dict[str, object] | None = None,
    expression_cfg: Dict[str, object] | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, str], np.ndarray]:
    ensure_mechinterp_path()
    from src.eval.gene_symbols import load_hgnc_alias_map, normalize_gene_names

    alias_map = load_hgnc_alias_map(paths.get("hgnc_alias_tsv"))
    gene_names_norm = normalize_gene_names(adata.var_names.values, alias_map)

    source_masks = []
    target_masks = []

    doro_source_mask, doro_target_mask = _candidate_masks_from_dorothea(
        gene_names_norm,
        paths,
        confidence_levels,
        candidate_cfg.use_sources_dorothea,
        candidate_cfg.use_targets_dorothea,
        alias_map,
    )
    source_masks.append(doro_source_mask)
    target_masks.append(doro_target_mask)

    omni_source_mask, omni_target_mask = _candidate_masks_from_omnipath(
        gene_names_norm,
        paths,
        candidate_cfg.use_sources_omnipath,
        candidate_cfg.use_targets_omnipath,
        alias_map,
        candidate_cfg.omnipath_directed_only,
        candidate_cfg.omnipath_exclude_underscore_composites,
    )
    source_masks.append(omni_source_mask)
    target_masks.append(omni_target_mask)

    intercell_cfg = intercell_cfg or {}
    intercell_source_mask, intercell_target_mask = _candidate_masks_from_intercell(
        gene_names_norm,
        paths,
        candidate_cfg.use_sources_intercell,
        candidate_cfg.use_targets_intercell,
        alias_map,
        intercell_cfg,
    )
    source_masks.append(intercell_source_mask)
    target_masks.append(intercell_target_mask)

    source_mask = _combine_masks(source_masks, candidate_cfg.candidate_mask_mode)
    target_mask = _combine_masks(target_masks, candidate_cfg.candidate_mask_mode)

    expression_cfg = expression_cfg or {}
    min_mean_expr = expression_cfg.get("min_mean_expr")
    min_frac_cells = expression_cfg.get("min_frac_cells")
    if min_mean_expr is not None or min_frac_cells is not None:
        expr_mask, _, _ = expression_masks(adata, min_mean_expr, min_frac_cells)
        apply_sources = bool(expression_cfg.get("apply_to_sources", True))
        apply_targets = bool(expression_cfg.get("apply_to_targets", True))
        if apply_sources:
            source_mask = expr_mask if source_mask is None else (source_mask & expr_mask)
        if apply_targets:
            target_mask = expr_mask if target_mask is None else (target_mask & expr_mask)

    return source_mask, target_mask, alias_map, gene_names_norm
