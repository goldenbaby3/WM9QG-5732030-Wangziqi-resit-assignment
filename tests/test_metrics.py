"""Tests for cost-sensitive threshold selection."""

import numpy as np

from maintainai.classification import threshold_table


def test_threshold_table_accounts_for_false_negative_cost() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.3, 0.8])
    table = threshold_table(y, probability, fn_cost=20.0, fp_cost=1.0)
    selected = table.loc[table["expected_cost"].idxmin()]
    assert selected["recall"] >= 0.5
    assert selected["expected_cost"] >= 0
