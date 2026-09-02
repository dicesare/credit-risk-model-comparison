# Historical experiment audit

## Scope inspected

Both `master` and `dev` contain the original notebook and a substantial V2 notebook. The latest V2 version has 33 cells, 53 outputs and 22 embedded figures. `projet_4_V2` contains no additional implementation.

## Breadth of work

The historical study includes iterative missing-value imputation, scaling, SMOTE, confusion matrices, ROC curves, threshold selection, five-fold grid search and comparisons across Logistic Regression, Elastic Net, Decision Tree, Random Forest, KNN and XGBoost.

## Verified results

- Logistic ROC: AUROC 0.729; at sensitivity 0.95, specificity 0.22, threshold 0.25 and F-measure 0.36.
- Logistic grid search: best CV AUROC 0.736234 with C=0.01, L2 and LBFGS.
- Elastic Net: best CV AUROC 0.736238 with C=0.01 and l1_ratio 0.4444.
- KNN: best CV AUROC 0.970152 with Manhattan distance and seven neighbours.
- XGBoost: best CV AUROC 0.976792 with colsample_bytree 0.9, learning rate 0.1, depth 15 and 100 trees.

## Audit findings

The notebook is valuable historical research, but several displayed scores cannot be promoted unchanged:

1. one tuned logistic cell prints accuracy from a prediction variable created by an earlier model;
2. Random Forest ROC cells call `model_log_reg.predict_proba`, so their displayed AUROC is not a Random Forest AUROC;
3. translated invalid solver names cause 50 failed grid-search fits;
4. resampling and cross-validation need a single imbalanced-learn pipeline to guarantee fold isolation;
5. very high KNN/XGBoost CV scores and large XGBoost fold variance require leakage and split audits;
6. accuracy near 90% is not sufficient for an imbalanced, consequential outcome.

## Current public implementation

The cleaned repository corrects the engineering pattern: one preprocessing/model pipeline, stratified separation, class weighting by default, synthetic tests, business-aware metrics and explicit governance. A future reproducible benchmark should place SMOTE inside each training fold and report calibration, subgroup performance and uncertainty across seeds.

This audit is intentionally visible: identifying methodological weaknesses in older work and rebuilding the protocol is evidence of current seniority, not something to conceal.

