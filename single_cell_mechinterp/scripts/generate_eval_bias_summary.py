from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False):
        values = [_format_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _optional_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [col for col in cols if col in df.columns]


def _compute_ci_width(df: pd.DataFrame, lower_col: str, upper_col: str, out_col: str) -> pd.DataFrame:
    if lower_col in df.columns and upper_col in df.columns:
        df = df.copy()
        df[out_col] = df[upper_col] - df[lower_col]
    return df


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_heatmap(data: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    values = data.values
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticklabels(data.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(path)


def _plot_bar_grouped(df: pd.DataFrame, x: str, hue: str, y: str, title: str, path: Path) -> None:
    categories = list(df[x].unique())
    hues = list(df[hue].unique())
    x_idx = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, hue_value in enumerate(hues):
        subset = df[df[hue] == hue_value]
        values = [subset[subset[x] == category][y].mean() for category in categories]
        positions = x_idx + (idx - (len(hues) - 1) / 2) * width
        ax.bar(positions, values, width=width, label=hue_value)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(title=hue)
    _save_fig(path)


def _plot_bar(data: pd.DataFrame, x: str, y: str, title: str, path: Path, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data[x], data[y])
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(data[x])))
    ax.set_xticklabels(data[x], rotation=30, ha="right")
    if log_y:
        ax.set_yscale("log")
    _save_fig(path)


def _plot_box(data: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    groups = [data[data[x] == label][y].values for label in data[x].unique()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(groups, labels=list(data[x].unique()), showfliers=True)
    ax.set_ylabel(y)
    ax.set_title(title)
    _save_fig(path)


def _plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    path: Path,
    x_log: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in data[group].unique():
        subset = data[data[group] == label]
        ax.scatter(subset[x], subset[y], label=label, alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    if x_log:
        ax.set_xscale("log")
    ax.legend(title=group)
    _save_fig(path)


def _plot_line(data: pd.DataFrame, x: str, y: str, group: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in data[group].unique():
        subset = data[data[group] == label].sort_values(x)
        ax.plot(subset[x], subset[y], marker="o", label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(title=group)
    _save_fig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary tables and plots for eval bias protocol")
    parser.add_argument("--input-dir", default="outputs/eval_bias")
    parser.add_argument("--output-dir", default="outputs/eval_bias")
    args = parser.parse_args()

    input_dir = _resolve(args.input_dir, PROJECT_ROOT)
    output_dir = _resolve(args.output_dir, PROJECT_ROOT)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    _ensure_dir(fig_dir)
    _ensure_dir(table_dir)

    coverage_path = input_dir / "coverage_tables.tsv"
    protocol_path = input_dir / "protocol_comparison.csv"
    noise_path = input_dir / "noise_stability_summary.tsv"
    mapping_path = input_dir / "symbol_mapping_report.tsv"

    if not coverage_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Missing coverage_tables.tsv or protocol_comparison.csv; run eval bias first.")

    coverage = pd.read_csv(coverage_path, sep="\t")
    protocol = pd.read_csv(protocol_path)
    noise = pd.read_csv(noise_path, sep="\t") if noise_path.exists() else pd.DataFrame()
    mapping = pd.read_csv(mapping_path, sep="\t", low_memory=False) if mapping_path.exists() else pd.DataFrame()

    # Coverage summary (mapped vs normalized)
    mapped = coverage[coverage["mapping_stage"] == "mapped"]
    normalized = coverage[coverage["mapping_stage"] == "normalized"]
    merge_keys = ["gene_set", "gold_standard"]
    if "mapping_policy" in coverage.columns:
        merge_keys.append("mapping_policy")
    coverage_summary = mapped.merge(
        normalized,
        on=merge_keys,
        suffixes=("_mapped", "_normalized"),
    )
    coverage_summary["edge_coverage_delta"] = (
        coverage_summary["edge_coverage_mapped"] - coverage_summary["edge_coverage_normalized"]
    )
    coverage_summary["gene_coverage_delta"] = (
        coverage_summary["gene_coverage_mapped"] - coverage_summary["gene_coverage_normalized"]
    )
    coverage_columns = [
        "gene_set",
        "gold_standard",
        "edge_coverage_mapped",
        "edge_coverage_normalized",
        "edge_coverage_delta",
        "gene_coverage_mapped",
        "gene_coverage_normalized",
        "gene_coverage_delta",
    ]
    if "mapping_policy" in coverage_summary.columns:
        coverage_columns.insert(2, "mapping_policy")
    coverage_summary = coverage_summary[coverage_columns]
    coverage_summary.to_csv(table_dir / "coverage_summary.csv", index=False)

    # Candidate set summary
    candidate_group_cols = _optional_cols(
        protocol, ["prediction_method", "mapping_policy", "gold_standard", "candidate_set"]
    )
    if not candidate_group_cols:
        candidate_group_cols = ["gold_standard", "candidate_set"]
    candidate_summary = (
        protocol.groupby(candidate_group_cols)
        .agg(
            aupr_median=("aupr", "median"),
            auroc_median=("auroc", "median"),
            f1_median=("f1", "median"),
            base_rate_median=("base_rate", "median"),
            candidate_size_median=("candidate_size", "median"),
        )
        .reset_index()
    )
    candidate_summary.to_csv(table_dir / "candidate_summary.csv", index=False)

    method_summary = pd.DataFrame()
    method_group_cols = _optional_cols(protocol, ["prediction_method", "mapping_policy"])
    if method_group_cols:
        method_summary = (
            protocol.groupby(method_group_cols)
            .agg(
                aupr_median=("aupr", "median"),
                auroc_median=("auroc", "median"),
                f1_median=("f1", "median"),
                base_rate_median=("base_rate", "median"),
            )
            .reset_index()
        )
        method_summary.to_csv(table_dir / "method_summary.csv", index=False)

    tissue_method_summary = pd.DataFrame()
    if "gene_set" in protocol.columns and "prediction_method" in protocol.columns:
        tissue_method_summary = (
            protocol.groupby(["gene_set", "prediction_method"])
            .agg(
                aupr_median=("aupr", "median"),
                auroc_median=("auroc", "median"),
                f1_median=("f1", "median"),
            )
            .reset_index()
        )
        tissue_method_summary.to_csv(table_dir / "method_summary_by_tissue.csv", index=False)

    tissue_candidate_summary = pd.DataFrame()
    if "gene_set" in protocol.columns and "candidate_set" in protocol.columns:
        tissue_candidate_summary = (
            protocol.groupby(["gene_set", "candidate_set"])
            .agg(
                aupr_median=("aupr", "median"),
                base_rate_median=("base_rate", "median"),
                candidate_size_median=("candidate_size", "median"),
            )
            .reset_index()
        )
        tissue_candidate_summary.to_csv(table_dir / "candidate_summary_by_tissue.csv", index=False)

    tissue_method_candidate = pd.DataFrame()
    if {"gene_set", "candidate_set", "prediction_method"}.issubset(protocol.columns):
        tissue_method_candidate = (
            protocol.groupby(["gene_set", "candidate_set", "prediction_method"])["aupr"]
            .median()
            .reset_index(name="aupr_median")
        )
        tissue_method_candidate.to_csv(table_dir / "method_candidate_by_tissue.csv", index=False)

    bootstrap_summary = pd.DataFrame()
    protocol_ci = _compute_ci_width(protocol, "aupr_ci_lower", "aupr_ci_upper", "aupr_ci_width")
    if "aupr_ci_width" in protocol_ci.columns:
        boot_group_cols = _optional_cols(protocol_ci, ["prediction_method", "mapping_policy", "candidate_set"])
        if not boot_group_cols:
            boot_group_cols = ["candidate_set"]
        bootstrap_summary = (
            protocol_ci.groupby(boot_group_cols)
            .agg(
                aupr_ci_width_median=("aupr_ci_width", "median"),
                aupr_ci_width_mean=("aupr_ci_width", "mean"),
            )
            .reset_index()
        )
        bootstrap_summary.to_csv(table_dir / "bootstrap_summary.csv", index=False)

    # Noise summary (random noise)
    noise_summary = pd.DataFrame()
    if not noise.empty:
        noise_random = noise[noise["noise_type"] == "random"]
        noise_group_cols = _optional_cols(
            noise_random, ["prediction_method", "mapping_policy", "candidate_set", "noise_rate"]
        )
        if not noise_group_cols:
            noise_group_cols = ["candidate_set", "noise_rate"]
        noise_summary = (
            noise_random.groupby(noise_group_cols)
            .agg(
                aupr_mean=("aupr_mean", "mean"),
                aupr_std=("aupr_std", "mean"),
                auroc_mean=("auroc_mean", "mean"),
            )
            .reset_index()
        )
        noise_summary.to_csv(table_dir / "noise_summary_random.csv", index=False)

    # Mapping summary
    if not mapping.empty:
        mapping_group_cols = _optional_cols(mapping, ["mapping_policy", "context", "status"])
        if not mapping_group_cols:
            mapping_group_cols = ["context", "status"]
        mapping_summary = mapping.groupby(mapping_group_cols).size().reset_index(name="count")
        mapping_summary.to_csv(table_dir / "mapping_summary.csv", index=False)
        mapping_summary["context_type"] = mapping_summary["context"].str.split(":", n=1).str[0]
        type_group_cols = _optional_cols(mapping_summary, ["mapping_policy", "context_type", "status"])
        if not type_group_cols:
            type_group_cols = ["context_type", "status"]
        mapping_summary_type = (
            mapping_summary.groupby(type_group_cols)["count"]
            .sum()
            .reset_index()
        )
        mapping_summary_type.to_csv(table_dir / "mapping_summary_by_type.csv", index=False)

    # Markdown summary
    markdown_sections = ["# Eval Bias Summary Tables", ""]
    markdown_sections.append("## Coverage Summary")
    markdown_sections.append(_df_to_markdown(coverage_summary))
    markdown_sections.append("")
    markdown_sections.append("## Candidate Set Summary")
    markdown_sections.append(_df_to_markdown(candidate_summary))
    markdown_sections.append("")
    if not method_summary.empty:
        markdown_sections.append("## Method Summary")
        markdown_sections.append(_df_to_markdown(method_summary))
        markdown_sections.append("")
    if not tissue_method_summary.empty:
        markdown_sections.append("## Method Summary by Tissue")
        markdown_sections.append(_df_to_markdown(tissue_method_summary))
        markdown_sections.append("")
    if not tissue_candidate_summary.empty:
        markdown_sections.append("## Candidate Set Summary by Tissue")
        markdown_sections.append(_df_to_markdown(tissue_candidate_summary))
        markdown_sections.append("")
    if not bootstrap_summary.empty:
        markdown_sections.append("## Bootstrap Summary")
        markdown_sections.append(_df_to_markdown(bootstrap_summary))
        markdown_sections.append("")
    if not mapping.empty:
        markdown_sections.append("## Mapping Summary (By Type)")
        markdown_sections.append(_df_to_markdown(mapping_summary_type))
        markdown_sections.append("")
    if not noise_summary.empty:
        markdown_sections.append("## Noise Summary (Random)")
        markdown_sections.append(_df_to_markdown(noise_summary))
        markdown_sections.append("")
    markdown_path = output_dir / "summary_tables.md"
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    # Plots
    heatmap_data = (
        mapped.groupby(["gene_set", "gold_standard"])["edge_coverage"]
        .mean()
        .unstack(fill_value=0.0)
    )
    _plot_heatmap(heatmap_data, "Edge Coverage (Mapped)", fig_dir / "edge_coverage_heatmap.png")

    gene_heatmap = (
        mapped.groupby(["gene_set", "gold_standard"])["gene_coverage"]
        .mean()
        .unstack(fill_value=0.0)
    )
    _plot_heatmap(gene_heatmap, "Gene Coverage (Mapped)", fig_dir / "gene_coverage_heatmap.png")

    coverage_stage = coverage.groupby(["gold_standard", "mapping_stage"]).agg(
        edge_coverage_mean=("edge_coverage", "mean")
    ).reset_index()
    _plot_bar_grouped(
        coverage_stage,
        x="gold_standard",
        hue="mapping_stage",
        y="edge_coverage_mean",
        title="Mean Edge Coverage by Mapping Stage",
        path=fig_dir / "edge_coverage_mapping_stage.png",
    )

    if "mapping_policy" in coverage.columns:
        policy_coverage = (
            coverage[coverage["mapping_stage"] == "mapped"]
            .groupby("mapping_policy")["edge_coverage"]
            .mean()
            .reset_index()
        )
        _plot_bar(
            policy_coverage,
            x="mapping_policy",
            y="edge_coverage",
            title="Mean Edge Coverage by Mapping Policy",
            path=fig_dir / "edge_coverage_by_policy.png",
        )

    _plot_box(
        protocol,
        x="candidate_set",
        y="aupr",
        title="AUPR by Candidate Set",
        path=fig_dir / "aupr_candidate_set_box.png",
    )

    candidate_overview = (
        protocol.groupby("candidate_set")
        .agg(
            base_rate_median=("base_rate", "median"),
            candidate_size_median=("candidate_size", "median"),
        )
        .reset_index()
    )
    _plot_bar(
        candidate_overview,
        x="candidate_set",
        y="candidate_size_median",
        title="Candidate Size (Median)",
        path=fig_dir / "candidate_size_bar.png",
        log_y=True,
    )
    _plot_bar(
        candidate_overview,
        x="candidate_set",
        y="base_rate_median",
        title="Base Rate (Median)",
        path=fig_dir / "base_rate_bar.png",
        log_y=True,
    )
    if not tissue_candidate_summary.empty:
        _plot_bar_grouped(
            tissue_candidate_summary,
            x="candidate_set",
            hue="gene_set",
            y="aupr_median",
            title="Median AUPR by Candidate Set and Tissue",
            path=fig_dir / "aupr_by_candidate_set_tissue.png",
        )

    if "prediction_method" in protocol.columns:
        method_aupr = (
            protocol.groupby("prediction_method")["aupr"]
            .median()
            .reset_index()
        )
        _plot_bar(
            method_aupr,
            x="prediction_method",
            y="aupr",
            title="Median AUPR by Method",
            path=fig_dir / "aupr_by_method.png",
            log_y=True,
        )
        method_heatmap = (
            protocol.groupby(["prediction_method", "candidate_set"])["aupr"]
            .median()
            .unstack(fill_value=0.0)
        )
        _plot_heatmap(
            method_heatmap,
            "Median AUPR by Method and Candidate Set",
            fig_dir / "aupr_method_candidate_heatmap.png",
        )
        if not tissue_method_summary.empty:
            _plot_bar_grouped(
                tissue_method_summary,
                x="prediction_method",
                hue="gene_set",
                y="aupr_median",
                title="Median AUPR by Method and Tissue",
                path=fig_dir / "aupr_by_method_tissue.png",
            )
        if not tissue_method_candidate.empty:
            subset = tissue_method_candidate[
                tissue_method_candidate["candidate_set"] == "tf_sources_targets"
            ]
            if not subset.empty:
                heatmap_tissue = (
                    subset.groupby(["gene_set", "prediction_method"])["aupr_median"]
                    .median()
                    .unstack(fill_value=0.0)
                )
                _plot_heatmap(
                    heatmap_tissue,
                    "Median AUPR by Method and Tissue (TF-Source-Target)",
                    fig_dir / "aupr_method_tissue_tf_sources_targets_heatmap.png",
                )

    if "mapping_policy" in protocol.columns:
        policy_aupr = (
            protocol.groupby(["mapping_policy", "candidate_set"])["aupr"]
            .median()
            .reset_index()
        )
        _plot_bar_grouped(
            policy_aupr,
            x="mapping_policy",
            hue="candidate_set",
            y="aupr",
            title="Median AUPR by Mapping Policy",
            path=fig_dir / "aupr_by_mapping_policy.png",
        )

    aupr_by_gold = candidate_summary.copy()
    if "prediction_method" in aupr_by_gold.columns or "mapping_policy" in aupr_by_gold.columns:
        group_cols = [col for col in ["gold_standard", "candidate_set"] if col in aupr_by_gold.columns]
        aupr_by_gold = (
            aupr_by_gold.groupby(group_cols)["aupr_median"]
            .median()
            .reset_index()
        )
    _plot_bar_grouped(
        aupr_by_gold,
        x="gold_standard",
        hue="candidate_set",
        y="aupr_median",
        title="Median AUPR by Gold Standard",
        path=fig_dir / "aupr_by_gold_standard.png",
    )

    _plot_scatter(
        protocol,
        x="base_rate",
        y="aupr",
        group="candidate_set",
        title="AUPR vs Base Rate",
        path=fig_dir / "aupr_base_rate_scatter.png",
    )

    _plot_scatter(
        protocol,
        x="candidate_size",
        y="aupr",
        group="candidate_set",
        title="AUPR vs Candidate Size",
        path=fig_dir / "aupr_candidate_size_scatter.png",
        x_log=True,
    )

    if not mapping.empty:
        mapping_plot = mapping_summary_type.copy()
        if "mapping_policy" in mapping_plot.columns:
            mapping_plot = (
                mapping_plot.groupby(["context_type", "status"])["count"]
                .sum()
                .reset_index()
            )
        _plot_bar_grouped(
            mapping_plot,
            x="context_type",
            hue="status",
            y="count",
            title="Mapping Status Counts by Context Type",
            path=fig_dir / "mapping_status_by_type.png",
        )

    if not bootstrap_summary.empty:
        if "prediction_method" in bootstrap_summary.columns:
            _plot_bar_grouped(
                bootstrap_summary,
                x="candidate_set",
                hue="prediction_method",
                y="aupr_ci_width_median",
                title="Median AUPR CI Width by Method",
                path=fig_dir / "aupr_ci_width_by_method.png",
            )
        else:
            _plot_bar(
                bootstrap_summary,
                x="candidate_set",
                y="aupr_ci_width_median",
                title="Median AUPR CI Width",
                path=fig_dir / "aupr_ci_width.png",
            )

    if not noise_summary.empty:
        noise_plot = noise_summary.copy()
        if "prediction_method" in noise_plot.columns or "mapping_policy" in noise_plot.columns:
            noise_plot = (
                noise_plot.groupby(["candidate_set", "noise_rate"])
                .agg(
                    aupr_mean=("aupr_mean", "mean"),
                    aupr_std=("aupr_std", "mean"),
                )
                .reset_index()
            )
        _plot_line(
            noise_plot,
            x="noise_rate",
            y="aupr_mean",
            group="candidate_set",
            title="AUPR vs Random Noise Rate",
            path=fig_dir / "aupr_noise_random.png",
        )
        _plot_line(
            noise_plot,
            x="noise_rate",
            y="aupr_std",
            group="candidate_set",
            title="AUPR Std vs Random Noise Rate",
            path=fig_dir / "aupr_noise_std_random.png",
        )


if __name__ == "__main__":
    main()
