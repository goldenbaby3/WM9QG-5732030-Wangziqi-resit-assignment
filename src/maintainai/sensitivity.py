"""Robustness checks for split strategy, random seed and operating cost."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .classification import classification_metrics, threshold_table
from .config import TARGET, ExperimentConfig
from .data import engineer_features, make_blocked_split
from .utils import ensure_dir, write_json


def run_sensitivity(
    frame: pd.DataFrame,
    classification_result: dict[str, object],
    config: ExperimentConfig,
) -> dict[str, object]:
    """Quantify how conclusions change under defensible alternative choices."""

    out = ensure_dir(config.artifact_dir / "sensitivity" / "tables")
    fitted_model = classification_result["model"]
    threshold = float(classification_result["threshold"])

    seed_rows = []
    seeds = [17, 101, 991] if config.quick else [17, 101, 991, 2027, 5732030]
    x_all = engineer_features(frame)
    y_all = frame[TARGET].astype(int)
    for seed in seeds:
        train_idx, test_idx = train_test_split(
            np.arange(len(frame)),
            test_size=config.test_size,
            random_state=seed,
            stratify=y_all,
        )
        model = clone(fitted_model)
        model.fit(x_all.iloc[train_idx], y_all.iloc[train_idx])
        probability = model.predict_proba(x_all.iloc[test_idx])[:, 1]
        metrics = classification_metrics(y_all.iloc[test_idx], probability, threshold)
        metrics["seed"] = seed
        seed_rows.append(metrics)
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(out / "random_seed_sensitivity.csv", index=False)

    x_train, x_block, y_train, y_block = make_blocked_split(frame, config.test_size)
    blocked_model = clone(fitted_model)
    blocked_model.fit(x_train, y_train)
    blocked_probability = blocked_model.predict_proba(x_block)[:, 1]
    blocked_metrics = classification_metrics(y_block, blocked_probability, threshold)
    blocked_metrics["test_failure_rate"] = float(y_block.mean())
    write_json(blocked_metrics, out / "blocked_split_metrics.json")

    y_test = classification_result["y_test"]
    probability = classification_result["test_probability"]
    cost_rows = []
    for fn_cost in [5.0, 10.0, 20.0, 50.0]:
        table = threshold_table(y_test, probability, fn_cost=fn_cost, fp_cost=1.0)
        best = table.loc[table["expected_cost"].idxmin()].to_dict()
        best["false_negative_cost"] = fn_cost
        cost_rows.append(best)
    pd.DataFrame(cost_rows).to_csv(out / "threshold_cost_sensitivity.csv", index=False)

    summary = {
        "seed_pr_auc_mean": float(seed_table["pr_auc"].mean()),
        "seed_pr_auc_std": float(seed_table["pr_auc"].std(ddof=1)),
        "seed_roc_auc_mean": float(seed_table["roc_auc"].mean()),
        "blocked_pr_auc": float(average_precision_score(y_block, blocked_probability)),
        "blocked_roc_auc": float(roc_auc_score(y_block, blocked_probability)),
        "blocked_failure_rate": float(y_block.mean()),
    }
    write_json(summary, out / "sensitivity_summary.json")
    return summary
