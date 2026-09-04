"""Command-line entry point for the complete evidence-generating workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .classification import run_classification
from .clustering import run_clustering
from .config import ExperimentConfig
from .data import load_dataset
from .eda import run_eda
from .explainability import run_local_explanation
from .sensitivity import run_sensitivity
from .utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Run reduced grids for verification."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config = ExperimentConfig(root=root, quick=args.quick)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Data validation and exploratory analysis")
    frame = load_dataset(config)
    audit = run_eda(frame, config)

    print("[2/5] Clustering comparison, stability and interpretation")
    clustering = run_clustering(frame, config)

    print("[3/5] Classification, tuning, calibration and locked-test evaluation")
    classification = run_classification(frame, config)

    print("[4/5] Global and local explainability")
    local = run_local_explanation(classification, config)

    print("[5/5] Robustness and cost-sensitivity analysis")
    sensitivity = run_sensitivity(frame, classification, config)

    write_json(
        {
            "data": audit,
            "clustering": {
                "algorithm": clustering["algorithm"],
                "clusters": clustering["clusters"],
            },
            "classification": classification["metrics"],
            "local_explanation": local,
            "sensitivity": sensitivity,
        },
        config.artifact_dir / "run_summary.json",
    )
    print(f"Completed: {config.artifact_dir}")


if __name__ == "__main__":
    main()
