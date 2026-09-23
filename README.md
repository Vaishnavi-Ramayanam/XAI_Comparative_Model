# XAI Comparative Model

A comparative study of machine-learning models and explainable AI techniques
for student academic performance analytics.

## Project structure

```text
data/       UCI Student Performance datasets
docs/       Project documentation
reports/    Generated evaluation and explainability graphs
results/    Application screenshots and final outputs
src/        Application code, training pipeline, dependencies, and model artifacts
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
