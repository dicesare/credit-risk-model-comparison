# Credit Risk Model Comparison

A reproducible comparison of interpretable and ensemble classifiers for imbalanced credit-risk data.

## Portfolio story

The project demonstrates how model selection changes when false negatives are more costly than false positives. It compares logistic regression and tree ensembles through a shared preprocessing pipeline, stratified validation and business-aware metrics.

## Highlights

- schema validation and leakage-resistant train/test separation;
- median imputation, scaling and one-hot encoding in a single pipeline;
- class weighting instead of opaque resampling by default;
- ROC-AUC, average precision, recall and confusion-matrix reporting;
- no customer data, pickle artifact or course document in the repository;
- synthetic fixtures for deterministic tests.

## Quick start

```bash
python -m venv .venv
pip install -e .[dev]
pytest
jupyter lab notebooks/model_comparison.ipynb
```

## Data

The original experiment used Home Credit-style application features. No source dataset is redistributed. Use an authorized copy of a clearly licensed dataset and record its version in `data/README.md`.

## Responsible modelling

This is an educational comparison, not a lending decision system. A production credit model requires legal review, fairness testing, explainability, monitoring and human governance.

## License

Code is released under the [MIT License](LICENSE); datasets retain their own terms.
