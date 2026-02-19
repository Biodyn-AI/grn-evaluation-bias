from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from network_inference.src.data.loaders import load_attention_scores, load_processed_anndata
from network_inference.src.inference.candidates import CandidateConfig, build_candidate_masks
from network_inference.src.utils.scm_imports import ensure_mechinterp_path


def infer_attention_network(config: Dict[str, object]) -> pd.DataFrame:
    paths = config["paths"]

    scores = load_attention_scores(paths["attention_scores"], paths["attention_counts"])
    adata = load_processed_anndata(paths["processed_h5ad"])

    network_cfg_data = dict(config.get("network", {}))
    candidate_cfg = CandidateConfig(
        use_sources_dorothea=bool(network_cfg_data.pop("candidate_sources_from_dorothea", False)),
        use_targets_dorothea=bool(network_cfg_data.pop("candidate_targets_from_dorothea", False)),
        use_sources_omnipath=bool(network_cfg_data.pop("candidate_sources_from_omnipath", False)),
        use_targets_omnipath=bool(network_cfg_data.pop("candidate_targets_from_omnipath", False)),
        use_sources_intercell=bool(network_cfg_data.pop("candidate_sources_from_intercell", False)),
        use_targets_intercell=bool(network_cfg_data.pop("candidate_targets_from_intercell", False)),
        candidate_mask_mode=network_cfg_data.pop("candidate_mask_mode", "union"),
        omnipath_directed_only=bool(config.get("omnipath", {}).get("directed_only", True)),
    )

    ensure_mechinterp_path()
    from src.network.export import export_edges_tsv, export_graphml
    from src.network.infer import NetworkConfig, infer_edges
    network_cfg = NetworkConfig(**network_cfg_data)
    source_mask, target_mask, _, _ = build_candidate_masks(
        adata,
        paths,
        candidate_cfg,
        config.get("evaluation", {}).get("dorothea_confidence"),
        intercell_cfg=config.get("intercell", {}),
        expression_cfg=config.get("expression_filter", {}),
    )

    edges = infer_edges(scores, adata.var_names, network_cfg, source_mask, target_mask)

    output_path = paths["network_edges"]
    export_edges_tsv(edges, output_path)

    graphml_path = paths.get("network_graphml")
    if graphml_path:
        export_graphml(edges, graphml_path)

    return edges
