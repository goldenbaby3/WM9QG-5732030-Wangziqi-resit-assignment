"""Exploratory data analysis with report-ready outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .config import (
    FAILURE_MODE_COLUMNS,
    ID_COLUMNS,
    TARGET,
    ExperimentConfig,
)
from .data import engineer_features
from .utils import ensure_dir, save_figure, soft_grid, write_json


def run_eda(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, object]:
    """Audit data quality, imbalance, ranges, relationships and row dependence."""

    out = ensure_dir(config.artifact_dir / "eda")
    tables = ensure_dir(out / "tables")
    figures = ensure_dir(out / "figures")

    x = engineer_features(frame)
    numeric = x.select_dtypes(include=np.number)
    target_counts = frame[TARGET].value_counts().sort_index()

    audit = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_udi": int(frame["UDI"].nunique()),
        "unique_product_id": int(frame["Product ID"].nunique()),
        "failure_count": int(target_counts.get(1, 0)),
        "failure_rate": float(frame[TARGET].mean()),
        "product_type_counts": frame["Type"].value_counts().to_dict(),
        "failure_mode_counts": frame[FAILURE_MODE_COLUMNS].sum().to_dict(),
        "forbidden_model_columns": ID_COLUMNS + FAILURE_MODE_COLUMNS,
    }
    write_json(audit, tables / "data_audit.json")

    quality_rows = []
    for column in frame.columns:
        series = frame[column]
        quality_rows.append(
            {
                "variable": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=False)),
                "minimum": float(series.min())
                if pd.api.types.is_numeric_dtype(series)
                else "",
                "maximum": float(series.max())
                if pd.api.types.is_numeric_dtype(series)
                else "",
            }
        )
    pd.DataFrame(quality_rows).to_csv(tables / "data_quality.csv", index=False)

    grouped = x.assign(**{TARGET: frame[TARGET]}).groupby(TARGET)
    grouped[numeric.columns].agg(["mean", "std", "median"]).to_csv(
        tables / "numeric_summary_by_outcome.csv"
    )

    corr_frame = x.copy()
    corr_frame["Type"] = corr_frame["Type"].map({"L": 0, "M": 1, "H": 2})
    corr_frame[TARGET] = frame[TARGET]
    corr = corr_frame.corr(method="spearman")
    corr.to_csv(tables / "spearman_correlations.csv")

    _plot_target(frame, figures / "target_distribution.png")
    _plot_numeric_distributions(x, frame[TARGET], figures / "feature_distributions.png")
    _plot_correlation(corr, figures / "spearman_heatmap.png")
    _plot_power_temperature(x, frame[TARGET], figures / "engineered_features.png")

    sequence = _sequence_diagnostics(frame)
    pd.DataFrame(sequence).to_csv(tables / "row_order_diagnostics.csv", index=False)
    return audit


def _plot_target(frame: pd.DataFrame, path) -> None:
    counts = frame[TARGET].value_counts().sort_index()
    labels = ["No failure", "Failure"]
    colours = ["#3B6EA8", "#D9772A"]
    plt.figure(figsize=(7, 4.2))
    bars = plt.bar(labels, counts.values, color=colours)
    for bar, count in zip(bars, counts.values, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            count + 100,
            f"{count:,}\n({count / len(frame):.1%})",
            ha="center",
        )
    plt.ylabel("Records")
    plt.title("Machine-failure class imbalance")
    plt.ylim(0, counts.max() * 1.13)
    soft_grid(plt.gca(), axis="y")
    save_figure(path)


def _plot_numeric_distributions(x: pd.DataFrame, y: pd.Series, path) -> None:
    columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature difference [K]",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, column in zip(axes.ravel(), columns, strict=True):
        for value, colour, label in [
            (0, "#3B6EA8", "No failure"),
            (1, "#D9772A", "Failure"),
        ]:
            ax.hist(
                x.loc[y == value, column],
                bins=28,
                density=True,
                alpha=0.48,
                color=colour,
                label=label,
            )
        ax.set_title(column)
        ax.set_ylabel("Density")
        soft_grid(ax, axis="y")
    axes[0, 0].legend()
    fig.suptitle("Operational distributions by machine-failure status", y=1.01)
    save_figure(path)


def _plot_correlation(corr: pd.DataFrame, path) -> None:
    plt.figure(figsize=(10, 8))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Spearman correlation")
    plt.xticks(range(len(corr)), corr.columns, rotation=70, ha="right", fontsize=8)
    plt.yticks(range(len(corr)), corr.index, fontsize=8)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(corr), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(corr), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.title("Spearman relationships (target shown for EDA only)")
    save_figure(path)


def _plot_power_temperature(x: pd.DataFrame, y: pd.Series, path) -> None:
    rng = np.random.default_rng(5732030)
    non_failure = x.index[y == 0]
    sampled_non_failure = rng.choice(
        non_failure, size=min(1500, len(non_failure)), replace=False
    )
    sample_idx = np.concatenate([sampled_non_failure, x.index[y == 1]])
    plt.figure(figsize=(8, 5.2))
    colours = np.where(y.loc[sample_idx].to_numpy() == 1, "#D1495B", "#5B8DB8")
    plt.scatter(
        x.loc[sample_idx, "Temperature difference [K]"],
        x.loc[sample_idx, "Mechanical power [W]"],
        c=colours,
        s=16,
        alpha=0.55,
    )
    plt.xlabel("Process minus air temperature [K]")
    plt.ylabel("Mechanical power proxy [W]")
    plt.title("Engineered operating features (failure cases highlighted)")
    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#5B8DB8",
                label="No failure",
                markersize=7,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#D1495B",
                label="Failure",
                markersize=7,
            ),
        ]
    )
    soft_grid(plt.gca())
    save_figure(path)


def _sequence_diagnostics(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    for column in [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        TARGET,
    ]:
        rows.append(
            {
                "variable": column,
                "lag_1_autocorrelation": float(frame[column].autocorr(lag=1)),
                "lag_10_autocorrelation": float(frame[column].autocorr(lag=10)),
            }
        )
    return rows
