"""Operating-state clustering with algorithm comparison and stability checks."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, ExperimentConfig
from .data import assert_predictor_safety, engineer_features
from .utils import ensure_dir, save_figure, soft_grid, write_json


def clustering_preprocessor() -> ColumnTransformer:
    """Scale all clustering inputs and one-hot encode product type."""

    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "product_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def run_clustering(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, object]:
    """Compare K-Means and Gaussian mixtures, then profile the selected solution."""

    out = ensure_dir(config.artifact_dir / "clustering")
    tables = ensure_dir(out / "tables")
    figures = ensure_dir(out / "figures")
    models = ensure_dir(out / "models")

    x = engineer_features(frame)
    assert_predictor_safety(list(x.columns))
    preprocess = clustering_preprocessor()
    transformed = preprocess.fit_transform(x)

    rng = np.random.default_rng(config.random_state)
    metric_sample_size = min(len(x), 2500 if config.quick else 5000)
    metric_idx = rng.choice(len(x), size=metric_sample_size, replace=False)
    rows = []
    max_k = 4 if config.quick else 6
    for algorithm in ["kmeans", "gaussian_mixture"]:
        for k in range(2, max_k + 1):
            estimator = _make_estimator(algorithm, k, config.random_state, config.quick)
            labels = estimator.fit_predict(transformed)
            sampled = transformed[metric_idx]
            sampled_labels = labels[metric_idx]
            row = {
                "algorithm": algorithm,
                "clusters": k,
                "silhouette": silhouette_score(sampled, sampled_labels),
                "calinski_harabasz": calinski_harabasz_score(sampled, sampled_labels),
                "davies_bouldin": davies_bouldin_score(sampled, sampled_labels),
                "minimum_cluster_fraction": float(
                    pd.Series(labels).value_counts(normalize=True).min()
                ),
                "stability_ari": _stability(
                    algorithm, k, transformed, labels, config.random_state, config.quick
                ),
            }
            if isinstance(estimator, GaussianMixture):
                row["bic"] = estimator.bic(transformed)
                row["aic"] = estimator.aic(transformed)
            else:
                row["bic"] = np.nan
                row["aic"] = np.nan
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison["silhouette_rank"] = comparison["silhouette"].rank(ascending=False)
    comparison["davies_rank"] = comparison["davies_bouldin"].rank(ascending=True)
    comparison["stability_rank"] = comparison["stability_ari"].rank(ascending=False)
    comparison["selection_rank_sum"] = comparison[
        ["silhouette_rank", "davies_rank", "stability_rank"]
    ].sum(axis=1)
    comparison = comparison.sort_values(
        ["selection_rank_sum", "silhouette"], ascending=[True, False]
    )
    comparison.to_csv(tables / "clustering_comparison.csv", index=False)

    best = comparison.iloc[0]
    algorithm = str(best["algorithm"])
    k = int(best["clusters"])
    estimator = _make_estimator(algorithm, k, config.random_state, config.quick)
    labels = estimator.fit_predict(transformed)

    profiles = _cluster_profiles(x, frame[TARGET], labels)
    profiles.to_csv(tables / "cluster_profiles.csv", index=False)
    sizes = pd.Series(labels).value_counts().sort_index()
    metadata = {
        "selected_algorithm": algorithm,
        "selected_clusters": k,
        "selection_rule": (
            "lowest combined rank across silhouette, Davies-Bouldin "
            "and seed stability"
        ),
        "model_inputs": list(x.columns),
        "target_used_to_form_clusters": False,
        "cluster_sizes": sizes.to_dict(),
        "failure_rate_used_post_hoc_only": True,
    }
    write_json(metadata, tables / "clustering_metadata.json")

    pca = PCA(n_components=2, random_state=config.random_state)
    projected = pca.fit_transform(transformed)
    write_json(
        {
            "pc1_explained_variance_ratio": pca.explained_variance_ratio_[0],
            "pc2_explained_variance_ratio": pca.explained_variance_ratio_[1],
            "total_two_component_variance": pca.explained_variance_ratio_.sum(),
        },
        tables / "pca_variance.json",
    )
    _plot_comparison(comparison, figures / "clustering_metric_comparison.png")
    _plot_pca(projected, labels, algorithm, k, figures / "cluster_pca_view.png", config)
    _plot_profiles(profiles, figures / "cluster_profile_heatmap.png")

    pipeline = Pipeline([("preprocess", preprocess), ("clusterer", estimator)])
    joblib.dump(pipeline, models / "clustering_pipeline.joblib")
    return {
        "algorithm": algorithm,
        "clusters": k,
        "labels": labels,
        "profiles": profiles,
    }


def _make_estimator(algorithm: str, k: int, seed: int, quick: bool):
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=10 if quick else 30, random_state=seed)
    if algorithm == "gaussian_mixture":
        return GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=2 if quick else 5,
            reg_covar=1e-6,
            random_state=seed,
        )
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def _stability(algorithm, k, transformed, reference_labels, seed, quick):
    scores = []
    repeats = 3 if quick else 6
    for offset in range(1, repeats + 1):
        candidate = _make_estimator(algorithm, k, seed + offset, quick)
        labels = candidate.fit_predict(transformed)
        scores.append(adjusted_rand_score(reference_labels, labels))
    return float(np.mean(scores))


def _cluster_profiles(x, target, labels):
    working = x.copy()
    working["cluster"] = labels
    working[TARGET] = target.to_numpy()
    rows = []
    overall = x[NUMERIC_FEATURES].mean()
    scale = x[NUMERIC_FEATURES].std(ddof=0).replace(0, 1)
    for cluster, group in working.groupby("cluster"):
        row = {
            "cluster": int(cluster),
            "records": len(group),
            "population_fraction": len(group) / len(working),
            "post_hoc_failure_rate": group[TARGET].mean(),
        }
        for product_type in ["L", "M", "H"]:
            row[f"type_{product_type}_fraction"] = (
                group["Type"] == product_type
            ).mean()
        for column in NUMERIC_FEATURES:
            row[f"mean_{column}"] = group[column].mean()
            row[f"standardised_difference_{column}"] = (
                group[column].mean() - overall[column]
            ) / scale[column]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


def _plot_comparison(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for algorithm, group in table.groupby("algorithm"):
        ordered = group.sort_values("clusters")
        axes[0].plot(
            ordered["clusters"], ordered["silhouette"], marker="o", label=algorithm
        )
        axes[1].plot(
            ordered["clusters"], ordered["davies_bouldin"], marker="o", label=algorithm
        )
        axes[2].plot(
            ordered["clusters"], ordered["stability_ari"], marker="o", label=algorithm
        )
    axes[0].set(title="Silhouette (higher is better)", xlabel="Clusters")
    axes[1].set(title="Davies-Bouldin (lower is better)", xlabel="Clusters")
    axes[2].set(title="Seed stability ARI (higher is better)", xlabel="Clusters")
    for ax in axes:
        ax.legend()
        soft_grid(ax, axis="y")
    save_figure(path)


def _plot_pca(projected, labels, algorithm, k, path, config):
    rng = np.random.default_rng(config.random_state)
    count = min(len(projected), 3000 if config.quick else 5000)
    idx = rng.choice(len(projected), size=count, replace=False)
    plt.figure(figsize=(8, 5.5))
    sampled_labels = np.asarray(labels)[idx]
    colours = plt.get_cmap("tab10")
    for cluster in range(k):
        mask = sampled_labels == cluster
        plt.scatter(
            projected[idx[mask], 0],
            projected[idx[mask], 1],
            color=colours(cluster),
            label=f"Cluster {cluster}",
            s=12,
            alpha=0.55,
        )
    plt.legend(markerscale=1.8)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title(f"Selected operating-state solution: {algorithm}, k={k}")
    soft_grid(plt.gca())
    save_figure(path)


def _plot_profiles(profiles, path):
    columns = [c for c in profiles.columns if c.startswith("standardised_difference_")]
    matrix = profiles.set_index("cluster")[columns]
    display = [c.replace("standardised_difference_", "") for c in columns]
    plt.figure(figsize=(12, max(3.2, len(matrix) * 0.8)))
    image = plt.imshow(matrix, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    plt.colorbar(image, label="Standardised mean difference")
    plt.xticks(range(len(display)), display, rotation=55, ha="right", fontsize=8)
    plt.yticks(range(len(matrix)), matrix.index)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(display), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xlabel("Operational feature")
    plt.ylabel("Cluster")
    plt.title("Interpretable operating-state profiles")
    save_figure(path)
