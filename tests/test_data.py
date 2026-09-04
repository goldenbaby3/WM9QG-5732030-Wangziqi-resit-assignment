"""Tests for the decisions most likely to invalidate the project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maintainai.config import FAILURE_MODE_COLUMNS, ID_COLUMNS, TARGET, ExperimentConfig
from maintainai.data import (
    assert_predictor_safety,
    engineer_features,
    make_locked_split,
)


def _frame(rows: int = 100) -> pd.DataFrame:
    failure = np.zeros(rows, dtype=int)
    failure[::10] = 1
    return pd.DataFrame(
        {
            "UDI": np.arange(rows),
            "Product ID": [f"L{i}" for i in range(rows)],
            "Type": ["L", "M"] * (rows // 2),
            "Air temperature [K]": np.full(rows, 300.0),
            "Process temperature [K]": np.full(rows, 310.0),
            "Rotational speed [rpm]": np.full(rows, 1500.0),
            "Torque [Nm]": np.full(rows, 40.0),
            "Tool wear [min]": np.arange(rows),
            TARGET: failure,
            **{name: np.zeros(rows, dtype=int) for name in FAILURE_MODE_COLUMNS},
        }
    )


def test_feature_engineering_uses_permitted_measurements() -> None:
    features = engineer_features(_frame())
    assert not set(ID_COLUMNS + [TARGET] + FAILURE_MODE_COLUMNS).intersection(features)
    assert np.allclose(features["Temperature difference [K]"], 10.0)
    expected_power = 40.0 * 1500.0 * 2.0 * np.pi / 60.0
    assert np.allclose(features["Mechanical power [W]"], expected_power)


def test_leakage_guard_rejects_failure_modes() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_predictor_safety(["Torque [Nm]", "TWF"])


def test_locked_split_has_no_index_overlap() -> None:
    config = ExperimentConfig(root=Path("."), test_size=0.2)
    x_dev, x_test, y_dev, y_test = make_locked_split(_frame(), config)
    assert x_dev.index.intersection(x_test.index).empty
    assert y_dev.index.equals(x_dev.index)
    assert y_test.index.equals(x_test.index)
