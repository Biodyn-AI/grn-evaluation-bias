from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.network.infer import NetworkConfig, infer_edges
from src.utils.config import load_config, resolve_path


def _resolve(path_value: str | Path, base_dir: Path) -> Path:
    return resolve_path(path_value, base_dir)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_datasets(cfg: Dict, base_dir: Path) -> List[Dict]:
    paths_cfg = cfg.get("paths", {})
    options_cfg = cfg.get("options", {})
    datasets_cfg = cfg.get("datasets")
    entries: List[Dict] = []

    if datasets_cfg:
        for dataset in datasets_cfg:
            name = dataset.get("name") or "dataset"
            input_value = dataset.get("input_h5ad") or paths_cfg.get("input_h5ad")
            if not input_value:
                raise ValueError(f"Dataset {name} is missing input_h5ad")
            base_output = paths_cfg.get("output_dir", "outputs/eval_bias/baselines")
            output_value = dataset.get("output_dir") or (Path(base_output) / name)
            output_dir = _resolve(output_value, base_dir)
            options = dict(options_cfg)
            options.update(dataset.get("options", {}) or {})
            entries.append(
                {
                    "name": name,
                    "input_h5ad": _resolve(input_value, base_dir),
                    "output_dir": output_dir,
                    "options": options,
                }
            )
    else:
        input_value = paths_cfg.get("input_h5ad")
        if not input_value:
            raise ValueError("paths.input_h5ad is required when datasets are not specified")
        output_value = paths_cfg.get("output_dir", "outputs/eval_bias/baselines")
        entries.append(
            {
                "name": "default",
                "input_h5ad": _resolve(input_value, base_dir),
                "output_dir": _resolve(output_value, base_dir),
                "options": dict(options_cfg),
            }
        )

    return entries


def _load_expression(path: Path, max_cells: int | None, max_genes: int | None, seed: int) -> tuple[np.ndarray, List[str]]:
    adata = sc.read_h5ad(path)
    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[indices].copy()
    if max_genes and adata.n_vars > max_genes:
        rng = np.random.default_rng(seed)
        indices = rng.choice(adata.n_vars, size=max_genes, replace=False)
        adata = adata[:, indices].copy()

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    gene_names = list(adata.var_names)
    return X, gene_names


def _subset_matrix(
    X: np.ndarray,
    gene_names: List[str],
    max_cells: int | None,
    max_genes: int | None,
    seed: int,
) -> tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    X_sub = X
    names_sub = list(gene_names)
    if max_cells and X_sub.shape[0] > max_cells:
        indices = rng.choice(X_sub.shape[0], size=max_cells, replace=False)
        X_sub = X_sub[indices, :]
    if max_genes and X_sub.shape[1] > max_genes:
        indices = rng.choice(X_sub.shape[1], size=max_genes, replace=False)
        X_sub = X_sub[:, indices]
        names_sub = [names_sub[idx] for idx in indices]
    return X_sub, names_sub


def _select_top_variance_indices(X: np.ndarray, indices: np.ndarray, max_count: int | None) -> np.ndarray:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    variances = np.var(X[:, indices], axis=0)
    order = np.argsort(-variances)
    return indices[order[:max_count]]


def _build_regressor_kwargs(
    base_kwargs: Dict,
    method_cfg: Dict,
    allowed_keys: Iterable[str],
) -> Dict:
    kwargs = dict(base_kwargs)
    for key in allowed_keys:
        if key in method_cfg and method_cfg[key] is not None:
            kwargs[key] = method_cfg[key]
    return kwargs


def _load_regulators(path: Path, gene_names: List[str]) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", dtype=str)
    columns = {col.lower(): col for col in df.columns}
    if "source" in columns:
        regulator_genes = set(df[columns["source"]].dropna().astype(str))
    elif "tf" in columns:
        regulator_genes = set(df[columns["tf"]].dropna().astype(str))
    else:
        regulator_genes = set(df.iloc[:, 0].dropna().astype(str))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    indices = [gene_to_idx[gene] for gene in regulator_genes if gene in gene_to_idx]
    return np.array(sorted(indices), dtype=int)


def _run_arboreto(
    X: np.ndarray,
    gene_names: List[str],
    tf_names: List[str],
    target_names: List[str],
    method_type: str,
    method_cfg: Dict,
    seed: int,
) -> pd.DataFrame:
    try:
        from arboreto.core import (
            EARLY_STOP_WINDOW_LENGTH,
            RF_KWARGS,
            SGBM_KWARGS,
            create_graph,
        )
        from distributed import Client, LocalCluster
    except ImportError as exc:
        raise ImportError("arboreto and dask are required for arboreto baselines") from exc

    if not tf_names:
        raise ValueError("No TF names remain after filtering regulators.")
    if not target_names:
        raise ValueError("No target genes remain after filtering targets.")

    method_key = method_type.lower()
    if method_key in {"arboreto_grnboost2", "grnboost2"}:
        regressor_type = "GBM"
        allowed_keys = {
            "learning_rate",
            "n_estimators",
            "max_features",
            "subsample",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        }
        regressor_kwargs = _build_regressor_kwargs(SGBM_KWARGS, method_cfg, allowed_keys)
    elif method_key in {"arboreto_genie3", "genie3"}:
        regressor_type = "RF"
        allowed_keys = {
            "n_estimators",
            "max_features",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "n_jobs",
        }
        regressor_kwargs = _build_regressor_kwargs(RF_KWARGS, method_cfg, allowed_keys)
    else:
        raise ValueError(f"Unknown arboreto method type: {method_type}")

    early_stop_window_length = int(method_cfg.get("early_stop_window_length", EARLY_STOP_WINDOW_LENGTH))
    method_seed = method_cfg.get("seed", seed)
    limit = method_cfg.get("limit")

    cluster_kwargs = {"diagnostics_port": None}
    if method_cfg.get("n_workers") is not None:
        cluster_kwargs["n_workers"] = int(method_cfg["n_workers"])
    if method_cfg.get("threads_per_worker") is not None:
        cluster_kwargs["threads_per_worker"] = int(method_cfg["threads_per_worker"])
    if method_cfg.get("memory_limit") is not None:
        cluster_kwargs["memory_limit"] = method_cfg["memory_limit"]

    cluster = LocalCluster(**cluster_kwargs)
    client = Client(cluster)
    try:
        graph = create_graph(
            expression_matrix=X,
            gene_names=gene_names,
            tf_names=tf_names,
            regressor_type=regressor_type,
            regressor_kwargs=regressor_kwargs,
            client=client,
            target_genes=target_names,
            limit=limit,
            include_meta=True,
            early_stop_window_length=early_stop_window_length,
            seed=method_seed,
        )
        edges_df, _ = client.compute(graph, sync=True)
    finally:
        client.close()
        cluster.close()
    return edges_df.sort_values(by="importance", ascending=False)


def _fit_regressor(
    X_reg: np.ndarray,
    y: np.ndarray,
    method: str,
    params: Dict,
    random_state: int,
):
    if method == "genie3":
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 50)),
            max_features=params.get("max_features", 0.3),
            max_depth=params.get("max_depth"),
            n_jobs=int(params.get("n_jobs", 1)),
            random_state=random_state,
        )
    elif method == "grnboost":
        model = GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown regression method: {method}")
    model.fit(X_reg, y)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError(f"Model {method} does not expose feature_importances_")
    return importances


def _pearson_correlation(X: np.ndarray) -> np.ndarray:
    n_cells = X.shape[0]
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    X = X / std
    corr = (X.T @ X) / max(n_cells - 1, 1)
    return corr


def _spearman_correlation(X: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
    except ImportError as exc:
        raise ImportError("scipy is required for spearman correlation") from exc

    ranks = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[1]):
        ranks[:, i] = rankdata(X[:, i], method="average")
    return _pearson_correlation(ranks)


def _mutual_info_matrix(X: np.ndarray, n_bins: int) -> np.ndarray:
    from sklearn.metrics import mutual_info_score

    n_cells, n_genes = X.shape
    discretized = np.zeros((n_cells, n_genes), dtype=np.int16)
    for idx in range(n_genes):
        values = X[:, idx]
        if np.all(values == values[0]):
            continue
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        quantiles = np.unique(quantiles)
        if quantiles.size <= 2:
            continue
        discretized[:, idx] = np.digitize(values, quantiles[1:-1], right=False)

    scores = np.zeros((n_genes, n_genes), dtype=np.float32)
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            mi_value = mutual_info_score(discretized[:, i], discretized[:, j])
            scores[i, j] = mi_value
            scores[j, i] = mi_value
    return scores


def _pidc_full_matrix(X: np.ndarray, n_bins: int) -> np.ndarray:
    mi = _mutual_info_matrix(X, n_bins)
    mi = mi.astype(np.float32, copy=False)
    n_genes = mi.shape[0]
    if n_genes == 0:
        return mi

    min_ij_k = np.minimum(mi[:, None, :], mi[None, :, :])
    sum_min = min_ij_k.sum(axis=2, dtype=np.float64)
    denom = max(n_genes - 2, 1)
    mean_min = sum_min / denom
    ui = mi - mean_min
    np.fill_diagonal(ui, 0.0)

    mean_i = ui.mean(axis=1, keepdims=True)
    std_i = ui.std(axis=1, keepdims=True, ddof=1)
    std_i[std_i == 0] = 1.0
    z = (ui - mean_i) / std_i
    z[z < 0] = 0.0
    scores = np.sqrt(z**2 + z.T**2)
    return scores


def _random_edges(
    gene_names: List[str],
    top_k: int,
    remove_self: bool,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    edges: List[tuple[str, str, float]] = []
    for idx, source in enumerate(gene_names):
        if top_k <= 0:
            continue
        targets = np.arange(n_genes)
        if remove_self:
            targets = targets[targets != idx]
        if targets.size == 0:
            continue
        k = min(top_k, targets.size)
        chosen = rng.choice(targets, size=k, replace=False)
        scores = rng.random(size=k)
        for target_idx, score in zip(chosen, scores):
            edges.append((source, gene_names[target_idx], float(score)))
    return pd.DataFrame(edges, columns=["source", "target", "score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline GRN predictions for eval bias")
    parser.add_argument("--config", default="configs/eval_bias_baselines.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    base_dir = PROJECT_ROOT

    seed = int(cfg.get("project", {}).get("seed", 42))
    dataset_entries = _collect_datasets(cfg, base_dir)

    for dataset in dataset_entries:
        input_path = dataset["input_h5ad"]
        output_dir = dataset["output_dir"]
        options_cfg = dataset["options"]
        _ensure_dir(output_dir)

        X, gene_names = _load_expression(
            input_path,
            max_cells=options_cfg.get("max_cells"),
            max_genes=options_cfg.get("max_genes"),
            seed=seed,
        )

        remove_self = bool(options_cfg.get("remove_self", True))

        for method in cfg.get("methods", []):
            name = method.get("name")
            if not name:
                continue
            method_type = method.get("type", "correlation")
            method_type_key = str(method_type).lower()
            top_k = int(method.get("top_k", 50))
            method_seed = int(method.get("seed", seed))
            X_method, gene_names_method = _subset_matrix(
                X,
                gene_names,
                method.get("max_cells"),
                method.get("max_genes"),
                method_seed,
            )
            if method_type == "random":
                edges = _random_edges(
                    gene_names_method,
                    top_k=top_k,
                    remove_self=remove_self,
                    seed=method_seed,
                )
                output_path = output_dir / f"{name}_edges.tsv"
                edges.to_csv(output_path, sep="\t", index=False)
                continue

            if method_type_key in {"arboreto_grnboost2", "arboreto_genie3", "grnboost2", "scenic_grnboost2"}:
                regulators_from = method.get("regulators_from")
                if regulators_from:
                    regulators_path = _resolve(regulators_from, base_dir)
                    regulator_indices = _load_regulators(regulators_path, gene_names_method)
                else:
                    regulator_indices = np.arange(len(gene_names_method))

                regulator_indices = _select_top_variance_indices(
                    X_method, regulator_indices, method.get("max_regulators")
                )
                target_indices = np.arange(len(gene_names_method))
                target_indices = _select_top_variance_indices(
                    X_method, target_indices, method.get("max_targets")
                )

                tf_names = [gene_names_method[idx] for idx in regulator_indices]
                target_names = [gene_names_method[idx] for idx in target_indices]
                edges_df = _run_arboreto(
                    X_method,
                    gene_names_method,
                    tf_names,
                    target_names,
                    "arboreto_grnboost2" if method_type_key == "scenic_grnboost2" else method_type,
                    method,
                    seed,
                )
                edges_df = edges_df.rename(columns={"TF": "source", "importance": "score"})
                if remove_self:
                    edges_df = edges_df[edges_df["source"] != edges_df["target"]]
                if top_k > 0 and not edges_df.empty:
                    edges_df = edges_df.sort_values(["target", "score"], ascending=[True, False])
                    edges_df = edges_df.groupby("target", as_index=False).head(top_k)
                output_path = output_dir / f"{name}_edges.tsv"
                edges_df.to_csv(output_path, sep="\t", index=False)
                continue

            if method_type in {"genie3", "grnboost"}:
                regulators_from = method.get("regulators_from")
                if regulators_from:
                    regulators_path = _resolve(regulators_from, base_dir)
                    regulator_indices = _load_regulators(regulators_path, gene_names_method)
                else:
                    regulator_indices = np.arange(len(gene_names_method))

                regulator_indices = _select_top_variance_indices(
                    X_method, regulator_indices, method.get("max_regulators")
                )
                target_indices = np.arange(len(gene_names_method))
                target_indices = _select_top_variance_indices(
                    X_method, target_indices, method.get("max_targets")
                )

                X_reg_full = X_method[:, regulator_indices]
                edges: List[tuple[str, str, float]] = []
                random_state = int(method.get("random_state", seed))
                for target_idx in target_indices:
                    y = X_method[:, target_idx]
                    X_reg = X_reg_full
                    regs = regulator_indices
                    if target_idx in regulator_indices:
                        mask = regulator_indices != target_idx
                        X_reg = X_reg_full[:, mask]
                        regs = regulator_indices[mask]
                    if X_reg.shape[1] == 0:
                        continue
                    importances = _fit_regressor(X_reg, y, method_type, method, random_state)
                    if importances.size == 0:
                        continue
                    k = min(top_k, importances.size)
                    top_idx = np.argpartition(-importances, k - 1)[:k]
                    for idx in top_idx:
                        source_idx = regs[idx]
                        edges.append(
                            (
                                gene_names_method[source_idx],
                                gene_names_method[target_idx],
                                float(importances[idx]),
                            )
                        )

                edges_df = pd.DataFrame(edges, columns=["source", "target", "score"])
                output_path = output_dir / f"{name}_edges.tsv"
                edges_df.to_csv(output_path, sep="\t", index=False)
                continue

            if method_type_key == "pidc_full":
                n_bins = int(method.get("bins", 10))
                scores = _pidc_full_matrix(X_method, n_bins=n_bins)
            elif method_type_key in {"pidc", "pidc_proxy"}:
                n_bins = int(method.get("bins", 10))
                scores = _mutual_info_matrix(X_method, n_bins=n_bins)
            else:
                corr_type = method.get("correlation", "pearson")
                if corr_type == "pearson":
                    scores = _pearson_correlation(X_method)
                elif corr_type == "spearman":
                    scores = _spearman_correlation(X_method)
                else:
                    raise ValueError(f"Unknown correlation type: {corr_type}")

                if method.get("abs_value", True):
                    scores = np.abs(scores)

            network_cfg = NetworkConfig(
                threshold_percentile=method.get("threshold_percentile", 99.0),
                top_k=top_k,
                remove_self=remove_self,
            )
            edges = infer_edges(scores, gene_names_method, network_cfg)
            output_path = output_dir / f"{name}_edges.tsv"
            edges.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
