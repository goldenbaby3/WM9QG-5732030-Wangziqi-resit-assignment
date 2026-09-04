"""Leakage-safe, imbalance-aware classification and probability evaluation."""

from __future__ import annotations

from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ExperimentConfig,
)
from .data import assert_predictor_safety, make_locked_split
from .utils import ensure_dir, save_figure, soft_grid, write_json


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing fitted only inside a pipeline or CV fold."""

    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "product_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidates(random_state: int, quick: bool = False) -> dict[str, Pipeline]:
    """Return interpretable linear and nonlinear alternatives plus a baseline."""

    trees = 180 if quick else 400
    return {
        "dummy_prior": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_balanced": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trees,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180 if quick else 300,
                        learning_rate=0.06,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def threshold_table(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> pd.DataFrame:
    """Evaluate operating thresholds using minority-class metrics and cost."""

    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": precision_score(y_true, prediction, zero_division=0),
                "recall": recall_score(y_true, prediction, zero_division=0),
                "f1": f1_score(y_true, prediction, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
                "expected_cost": float(fn_cost * fn + fp_cost * fp),
                "cost_per_record": float((fn_cost * fn + fp_cost * fp) / len(y_true)),
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute metrics that remain meaningful under 3.39% prevalence."""

    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_failure": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall_failure": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_failure": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def run_classification(
    frame: pd.DataFrame, config: ExperimentConfig
) -> dict[str, object]:
    """Compare, tune, calibrate and evaluate a locked-test classifier."""

    out = ensure_dir(config.artifact_dir / "classification")
    tables = ensure_dir(out / "tables")
    figures = ensure_dir(out / "figures")
    models_dir = ensure_dir(out / "models")

    x_dev, x_test, y_dev, y_test = make_locked_split(frame, config)
    assert_predictor_safety(list(x_dev.columns))
    split_table = pd.DataFrame(
        [
            {
                "split": "development",
                "records": len(y_dev),
                "failures": int(y_dev.sum()),
                "failure_rate": y_dev.mean(),
            },
            {
                "split": "locked_test",
                "records": len(y_test),
                "failures": int(y_test.sum()),
                "failure_rate": y_test.mean(),
            },
        ]
    )
    split_table.to_csv(tables / "data_split.csv", index=False)

    cv = StratifiedKFold(
        n_splits=config.cv_splits, shuffle=True, random_state=config.random_state
    )
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "f1": "f1",
    }
    candidates = build_candidates(config.random_state, config.quick)
    comparison_rows = []
    for name, estimator in candidates.items():
        result = cross_validate(
            estimator,
            x_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row: dict[str, object] = {"model": name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1))
        row["mean_fit_seconds"] = float(result["fit_time"].mean())
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "mean_pr_auc", ascending=False
    )
    comparison.to_csv(tables / "model_comparison_cv.csv", index=False)

    tuned_searches = _tune_models(candidates, x_dev, y_dev, cv, config)
    search_rows = []
    for name, search in tuned_searches.items():
        search_rows.append(
            {
                "model": name,
                "best_cv_pr_auc": float(search.best_score_),
                "best_parameters": str(search.best_params_),
            }
        )
        detail = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        detail.to_csv(tables / f"hyperparameter_search_{name}.csv", index=False)
    search_summary = pd.DataFrame(search_rows).sort_values(
        "best_cv_pr_auc", ascending=False
    )
    search_summary.to_csv(tables / "hyperparameter_search_summary.csv", index=False)

    best_name = str(search_summary.iloc[0]["model"])
    best_estimator = clone(tuned_searches[best_name].best_estimator_)

    rfecv_table = _run_rfecv(x_dev, y_dev, cv)
    rfecv_table.to_csv(tables / "rfecv_feature_selection.csv", index=False)

    calibration_method, calibration_summary = _select_calibration(
        best_estimator, x_dev, y_dev, config
    )
    calibration_summary.to_csv(tables / "calibration_method_selection.csv", index=False)

    calibrated = CalibratedClassifierCV(
        estimator=clone(best_estimator), method=calibration_method, cv=cv
    )
    oof_probability = cross_val_predict(
        calibrated,
        x_dev,
        y_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    thresholds = threshold_table(
        y_dev.to_numpy(),
        oof_probability,
        config.false_negative_cost,
        config.false_positive_cost,
    )
    thresholds.to_csv(tables / "threshold_selection_oof.csv", index=False)
    selected_threshold = float(
        thresholds.loc[thresholds["expected_cost"].idxmin(), "threshold"]
    )

    calibrated.fit(x_dev, y_dev)
    test_probability = calibrated.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, test_probability, selected_threshold)
    metrics.update(
        {
            "selected_model": best_name,
            "calibration_method": calibration_method,
            "false_negative_cost": config.false_negative_cost,
            "false_positive_cost": config.false_positive_cost,
            "prevalence_test": float(y_test.mean()),
            "config": asdict(config),
        }
    )
    write_json(metrics, tables / "locked_test_metrics.json")

    predictions = pd.DataFrame(
        {
            "row_index": x_test.index,
            "actual_failure": y_test.to_numpy(),
            "predicted_probability": test_probability,
            "predicted_failure": (test_probability >= selected_threshold).astype(int),
        }
    )
    predictions.to_csv(tables / "locked_test_predictions.csv", index=False)

    subgroup = _subgroup_metrics(x_test, y_test, test_probability, selected_threshold)
    subgroup.to_csv(tables / "subgroup_performance.csv", index=False)

    _plot_curves(y_test, test_probability, figures / "roc_precision_recall.png")
    _plot_confusion(
        y_test, test_probability, selected_threshold, figures / "confusion_matrix.png"
    )
    calibration_bins = _calibration_bins(y_test, test_probability)
    calibration_bins.to_csv(tables / "locked_test_calibration_bins.csv", index=False)
    _plot_calibration(calibration_bins, figures / "calibration_curve.png")
    _plot_thresholds(
        thresholds, selected_threshold, figures / "threshold_tradeoffs.png"
    )

    importance = permutation_importance(
        calibrated,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=config.permutation_repeats,
        random_state=config.random_state,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(tables / "permutation_importance.csv", index=False)
    _plot_importance(importance_table, figures / "permutation_importance.png")

    joblib.dump(calibrated, models_dir / "calibrated_classifier.joblib")
    return {
        "model": calibrated,
        "model_name": best_name,
        "threshold": selected_threshold,
        "metrics": metrics,
        "x_dev": x_dev,
        "x_test": x_test,
        "y_dev": y_dev,
        "y_test": y_test,
        "test_probability": test_probability,
    }


def _tune_models(candidates, x, y, cv, config):
    grids = {
        "logistic_balanced": {
            "model__C": [0.05, 0.2, 1.0, 5.0],
        },
        "random_forest_balanced": {
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 3, 8],
            "model__max_features": ["sqrt", 0.7],
        },
    }
    if config.quick:
        grids["logistic_balanced"]["model__C"] = [0.2, 1.0]
        grids["random_forest_balanced"] = {
            "model__max_depth": [8, None],
            "model__min_samples_leaf": [1, 5],
            "model__max_features": ["sqrt"],
        }
    searches = {}
    for name, grid in grids.items():
        search = GridSearchCV(
            candidates[name],
            grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(x, y)
        searches[name] = search
    return searches


def _run_rfecv(x, y, cv) -> pd.DataFrame:
    preprocess = build_preprocessor()
    transformed = preprocess.fit_transform(x)
    names = preprocess.get_feature_names_out()
    selector = RFECV(
        estimator=LogisticRegression(
            class_weight="balanced", max_iter=2500, random_state=5732030
        ),
        step=1,
        min_features_to_select=3,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )
    selector.fit(transformed, y)
    return pd.DataFrame(
        {
            "transformed_feature": names,
            "selected": selector.support_,
            "rank": selector.ranking_,
        }
    ).sort_values(["selected", "rank"], ascending=[False, True])


def _select_calibration(estimator, x, y, config):
    outer = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=config.random_state + 1
    )
    inner = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=config.random_state + 2
    )
    rows = []
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(
            estimator=clone(estimator), method=method, cv=inner
        )
        probability = cross_val_predict(
            calibrated, x, y, cv=outer, method="predict_proba", n_jobs=-1
        )[:, 1]
        rows.append(
            {
                "method": method,
                "oof_brier_score": brier_score_loss(y, probability),
                "oof_log_loss": log_loss(y, probability, labels=[0, 1]),
                "oof_pr_auc": average_precision_score(y, probability),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["oof_brier_score", "oof_log_loss"])
    return str(summary.iloc[0]["method"]), summary


def _subgroup_metrics(x, y, probability, threshold):
    working = x.copy()
    working["actual"] = y
    working["probability"] = probability
    working["wear_band"] = pd.qcut(
        working["Tool wear [min]"], q=3, labels=["low", "medium", "high"]
    )
    rows = []
    for variable in ["Type", "wear_band"]:
        for value, group in working.groupby(variable, observed=True):
            if group["actual"].nunique() < 2:
                continue
            values = classification_metrics(
                group["actual"], group["probability"], threshold
            )
            values.update(
                {
                    "subgroup_variable": variable,
                    "subgroup": str(value),
                    "records": len(group),
                }
            )
            rows.append(values)
    return pd.DataFrame(rows)


def _plot_curves(y, probability, path):
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y, probability):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(
        xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve"
    )
    axes[0].legend()
    soft_grid(axes[0])
    axes[1].plot(
        recall,
        precision,
        label=f"PR-AUC = {average_precision_score(y, probability):.3f}",
    )
    axes[1].axhline(
        y.mean(), color="black", linestyle="--", label=f"Prevalence = {y.mean():.3f}"
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend()
    soft_grid(axes[1])
    save_figure(path)


def _plot_confusion(y, probability, threshold, path):
    matrix = confusion_matrix(y, probability >= threshold, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.6))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["No failure", "Failure"])
    plt.yticks([0, 1], ["No failure", "Failure"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=13)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Locked-test confusion matrix (threshold = {threshold:.2f})")
    save_figure(path)


def _calibration_bins(y, probability):
    # Adaptive boundaries retain resolution near zero without discarding the
    # small number of high-risk predictions in this highly imbalanced problem.
    edges = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0])
    working = pd.DataFrame({"actual": np.asarray(y), "probability": probability})
    working["bin"] = pd.cut(
        working["probability"], bins=edges, include_lowest=True, right=True
    )
    rows = []
    for interval, group in working.groupby("bin", observed=True):
        rows.append(
            {
                "lower_bound": float(interval.left),
                "upper_bound": float(interval.right),
                "records": len(group),
                "mean_predicted_probability": group["probability"].mean(),
                "observed_failure_fraction": group["actual"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot_calibration(table, path):
    plt.figure(figsize=(6, 5))
    plt.plot(
        table["mean_predicted_probability"],
        table["observed_failure_fraction"],
        marker="o",
        label="Calibrated model",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed failure fraction")
    plt.title("Probability calibration on the locked test set")
    plt.legend()
    soft_grid(plt.gca())
    save_figure(path)


def _plot_thresholds(table, selected, path):
    plt.figure(figsize=(8, 5))
    plt.plot(table["threshold"], table["precision"], label="Precision")
    plt.plot(table["threshold"], table["recall"], label="Recall")
    plt.plot(table["threshold"], table["f1"], label="F1")
    plt.axvline(
        selected, color="black", linestyle="--", label=f"Selected = {selected:.2f}"
    )
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("Out-of-fold threshold trade-offs")
    plt.legend()
    soft_grid(plt.gca(), axis="y")
    save_figure(path)


def _plot_importance(table, path):
    shown = table.sort_values("importance_mean").tail(12)
    plt.figure(figsize=(8, 5.5))
    plt.barh(
        shown["feature"],
        shown["importance_mean"],
        xerr=shown["importance_std"],
        color="#4C78A8",
        alpha=0.9,
    )
    plt.xlabel("Decrease in test PR-AUC after permutation")
    plt.title("Global permutation importance")
    soft_grid(plt.gca(), axis="x")
    save_figure(path)
