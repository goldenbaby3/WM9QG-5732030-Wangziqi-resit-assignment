"""Local model-agnostic explanation for one high-risk test case."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .classification import build_preprocessor
from .config import (
    CATEGORICAL_FEATURES,
    RAW_FEATURES,
    ExperimentConfig,
)
from .utils import ensure_dir, save_figure, soft_grid, write_json


def run_local_explanation(
    classification_result: dict[str, object], config: ExperimentConfig
) -> dict[str, object]:
    """Fit a weighted local surrogate around a high-risk locked-test case."""

    out = ensure_dir(config.artifact_dir / "explainability")
    tables = ensure_dir(out / "tables")
    figures = ensure_dir(out / "figures")

    model = classification_result["model"]
    x_dev: pd.DataFrame = classification_result["x_dev"]
    x_test: pd.DataFrame = classification_result["x_test"]
    y_test: pd.Series = classification_result["y_test"]
    probability: np.ndarray = classification_result["test_probability"]

    failure_positions = np.flatnonzero(y_test.to_numpy() == 1)
    if len(failure_positions):
        # A correctly identified but non-saturated failure is more informative than
        # an almost-certain prediction, for which a local linear surrogate is flat.
        local_position = failure_positions[
            np.argmin(np.abs(probability[failure_positions] - 0.50))
        ]
    else:
        local_position = int(np.argmax(probability))
    case = x_test.iloc[[local_position]].copy()
    case_index = int(case.index[0])

    perturbations = _perturb(case, x_dev, config)
    model_probability = model.predict_proba(perturbations)[:, 1]

    transformer = build_preprocessor()
    transformed_dev = transformer.fit_transform(x_dev)
    transformed = transformer.transform(perturbations)
    transformed_case = transformer.transform(case)[0]
    feature_names = transformer.get_feature_names_out()

    scale = np.std(transformed_dev, axis=0, ddof=0)
    scale[scale == 0] = 1.0
    distances = np.sqrt(np.sum(((transformed - transformed_case) / scale) ** 2, axis=1))
    kernel_width = 0.50 * np.sqrt(transformed.shape[1])
    weights = np.exp(-((distances / kernel_width) ** 2))

    surrogate = Ridge(alpha=0.1)
    surrogate.fit(transformed, model_probability, sample_weight=weights)
    surrogate_probability = surrogate.predict(transformed)
    fidelity = r2_score(model_probability, surrogate_probability, sample_weight=weights)
    weighted_mean = np.average(transformed, axis=0, weights=weights)
    contributions = surrogate.coef_ * (transformed_case - weighted_mean)

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "surrogate_coefficient": surrogate.coef_,
            "case_transformed_value": transformed_case,
            "local_contribution": contributions,
            "absolute_contribution": np.abs(contributions),
        }
    ).sort_values("absolute_contribution", ascending=False)
    explanation.to_csv(tables / "local_surrogate_explanation.csv", index=False)

    case_values = case.iloc[0].to_dict()
    metadata = {
        "case_row_index": case_index,
        "actual_failure": int(y_test.iloc[local_position]),
        "model_probability": float(probability[local_position]),
        "surrogate_weighted_r2": float(fidelity),
        "perturbation_samples": config.local_samples,
        "case_values": case_values,
        "warning": (
            "Local surrogate coefficients describe this neighbourhood, "
            "not causal effects."
        ),
    }
    write_json(metadata, tables / "local_case_metadata.json")
    _plot_local(explanation, fidelity, figures / "local_surrogate_explanation.png")
    return metadata


def _perturb(case: pd.DataFrame, development: pd.DataFrame, config: ExperimentConfig):
    rng = np.random.default_rng(config.random_state)
    n = config.local_samples
    perturbed = pd.DataFrame(index=range(n), columns=development.columns)
    # Perturb only measured inputs, then recompute engineered features so that
    # local samples remain physically coherent (power must match speed × torque).
    for column in RAW_FEATURES[1:]:
        centre = float(case.iloc[0][column])
        sigma = float(development[column].std(ddof=0)) * 0.15
        values = rng.normal(centre, max(sigma, 1e-9), size=n)
        perturbed[column] = np.clip(
            values, development[column].min(), development[column].max()
        )
    for column in CATEGORICAL_FEATURES:
        probabilities = development[column].value_counts(normalize=True)
        sampled = rng.choice(probabilities.index, size=n, p=probabilities.values)
        keep_case = rng.random(n) < 0.70
        sampled[keep_case] = case.iloc[0][column]
        perturbed[column] = sampled
    perturbed["Temperature difference [K]"] = (
        perturbed["Process temperature [K]"] - perturbed["Air temperature [K]"]
    )
    angular_speed = perturbed["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0)
    perturbed["Mechanical power [W]"] = perturbed["Torque [Nm]"] * angular_speed
    perturbed.iloc[0] = case.iloc[0]
    return perturbed.astype(development.dtypes.to_dict())


def _plot_local(table, fidelity, path):
    shown = table.head(10).sort_values("local_contribution")
    colours = np.where(shown["local_contribution"] >= 0, "#D1495B", "#3B6EA8")
    plt.figure(figsize=(8.5, 5.5))
    plt.barh(shown["feature"], shown["local_contribution"], color=colours)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Local contribution toward predicted failure probability")
    plt.title(f"Weighted local surrogate (neighbourhood R² = {fidelity:.3f})")
    soft_grid(plt.gca(), axis="x")
    save_figure(path)
