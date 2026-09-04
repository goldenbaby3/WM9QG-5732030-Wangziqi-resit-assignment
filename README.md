# MaintainAI predictive-maintenance project

This repository implements the WM9QG resit scenario using the UCI AI4I 2020
Predictive Maintenance Dataset. It is designed around reproducibility,
leakage prevention, class-imbalance-aware evaluation and auditable outputs.

## Analytical safeguards

- `UDI` and `Product ID` are identifiers and never enter a model.
- `TWF`, `HDF`, `PWF`, `OSF` and `RNF` are outcome-generating failure-mode
  labels and are excluded from every predictor set.
- `Machine failure` is excluded from clustering and inspected only after the
  clusters have been formed.
- Preprocessing and feature selection are fitted inside cross-validation.
- The locked test split is used only after model and threshold selection.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python -m ruff check src tests
python -m maintainai.run_all
```

For a faster verification run:

```powershell
python -m maintainai.run_all --quick
```

All tables, figures and fitted models are written to `artifacts/full` or
`artifacts/quick`. Generated outputs are evidence, not hand-edited results.

The full workflow exports data-quality audits, row-order diagnostics, PCA,
K-Means/Gaussian-mixture comparisons, cluster stability and profiles, four
classification baselines, tuned alternatives, RFECV evidence, nested
calibration selection, out-of-fold threshold selection, locked-test metrics,
subgroup performance, global permutation importance, a weighted local
surrogate and random-seed/blocked-split/cost sensitivity checks.

## Dataset citation

AI4I 2020 Predictive Maintenance Dataset. (2020). UCI Machine Learning
Repository. https://doi.org/10.24432/C5HS5C
