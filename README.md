# Explainable Student Failure Prediction

This project compares standalone and hybrid learning models for early student-failure prediction using the UCI Student Performance Mathematics and Portuguese datasets. The target is final-grade failure (`G3 < 10`), while `G3`, `G1`, and `G2` are excluded from predictors to support early intervention and prevent leakage.

## Implemented model groups

- Standalone baselines: Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting, and CatBoost.
- Heterogeneous soft-voting and stacking ensembles.
- ML + DL: Random Forest combined with a multilayer neural network.
- DL + DL: soft voting between two multilayer neural networks.
- DL + optimization: a neural network tuned with training-only grid search.
- ML + optimization: a Random Forest tuned with training-only grid search.

Every candidate uses the same preprocessing and held-out test split. Decision thresholds are selected from five-fold out-of-fold training predictions. Reported metrics include Accuracy, Precision, Recall, Specificity, F1-score, Balanced Accuracy, MCC, ROC-AUC, and PR-AUC.

## Project structure

```text
data/       Prescribed UCI Mathematics and Portuguese CSV files
docs/       Revised research paper
reports/    Generated comparison, evaluation, and SHAP figures
results/    Reserved for additional exported results
src/        Training pipeline, dashboard, dependencies, and trained artifacts
```

## Run the dashboard

```bash
pip install -r src/requirements.txt
streamlit run src/app.py
```

## Retrain the models

```bash
python src/train.py
```

Training reads the local dataset from `data/`, writes deployable model files to
`src/artifacts/`, and saves all generated graphs to `reports/`.

## Selected hybrid

The deployed model is the Logistic Regression + Random Forest + Histogram Gradient Boosting soft-voting ensemble. It achieved the strongest recall/F1-weighted deployment score among the evaluated hybrid candidates. The DL + DL ensemble achieved the highest ROC-AUC, so the results represent a metric trade-off rather than universal dominance by one model.
