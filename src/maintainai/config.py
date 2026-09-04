"""Central configuration for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TARGET = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = RAW_FEATURES[1:] + [
    "Temperature difference [K]",
    "Mechanical power [W]",
]
CATEGORICAL_FEATURES = ["Type"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by all analysis stages."""

    root: Path
    quick: bool = False
    random_state: int = 5732030
    test_size: float = 0.20
    false_negative_cost: float = 20.0
    false_positive_cost: float = 1.0

    @property
    def data_path(self) -> Path:
        return self.root / "data" / "raw" / "ai4i2020.csv"

    @property
    def artifact_dir(self) -> Path:
        return self.root / "artifacts" / ("quick" if self.quick else "full")

    @property
    def cv_splits(self) -> int:
        return 3 if self.quick else 5

    @property
    def permutation_repeats(self) -> int:
        return 5 if self.quick else 20

    @property
    def local_samples(self) -> int:
        return 500 if self.quick else 2500
