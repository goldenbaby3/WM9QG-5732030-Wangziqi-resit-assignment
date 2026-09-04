"""Data loading, validation, feature engineering and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    FAILURE_MODE_COLUMNS,
    ID_COLUMNS,
    RAW_FEATURES,
    TARGET,
    ExperimentConfig,
)

EXPECTED_COLUMNS = ID_COLUMNS + RAW_FEATURES + [TARGET] + FAILURE_MODE_COLUMNS


def load_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Load and validate the exact AI4I CSV used by the coursework."""

    frame = pd.read_csv(config.data_path)
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Fail fast on schema, target or leakage-related data changes."""

    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset schema. Expected the official 14-column AI4I file."
        )
    if len(frame) != 10_000:
        raise ValueError(f"Expected 10,000 rows; found {len(frame):,}.")
    if frame.isna().any().any():
        raise ValueError("The supplied AI4I file should contain no missing values.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Machine failure must be binary with values 0 and 1.")
    if frame["UDI"].duplicated().any() or frame["Product ID"].duplicated().any():
        raise ValueError("Identifiers must be unique in the supplied dataset.")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create physically interpretable features from permitted measurements."""

    features = frame[RAW_FEATURES].copy()
    features["Temperature difference [K]"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    angular_speed = features["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0)
    features["Mechanical power [W]"] = features["Torque [Nm]"] * angular_speed
    return features


def make_locked_split(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified development/test split."""

    indices = np.arange(len(frame))
    dev_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame[TARGET],
    )
    x = engineer_features(frame)
    y = frame[TARGET].astype(int)
    return x.iloc[dev_idx], x.iloc[test_idx], y.iloc[dev_idx], y.iloc[test_idx]


def make_blocked_split(
    frame: pd.DataFrame, test_size: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preserve row order for a sequence-dependence sensitivity check."""

    cut = int(len(frame) * (1.0 - test_size))
    x = engineer_features(frame)
    y = frame[TARGET].astype(int)
    return x.iloc[:cut], x.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def assert_predictor_safety(columns: list[str]) -> None:
    """Reject identifiers, target and outcome-generating failure-mode labels."""

    forbidden = set(ID_COLUMNS + [TARGET] + FAILURE_MODE_COLUMNS)
    leaked = forbidden.intersection(columns)
    if leaked:
        raise ValueError(f"Forbidden leakage columns detected: {sorted(leaked)}")
