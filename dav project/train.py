"""
train.py — Explainable Student Failure Prediction System
Final-year Data Science project.

Predicts whether a student is at risk of failing.
Produces artifacts/ with model, preprocessor, metadata, SHAP plots, and metrics.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay, balanced_accuracy_score,
    matthews_corrcoef, average_precision_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# UCI Student Performance: G3 is the final grade (0–20).
# Pass threshold: >= 10 (Portuguese grading scale, 10/20 is passing).
# fail = 1 means G3 < 10 (at risk of failing).
PASS_THRESHOLD = 10

# Whether to include G1/G2 as predictors.
# For realistic early-warning prediction, set to False.
# If True, the model will use period grades and be less useful for early intervention.
USE_PERIOD_GRADES = False

# Small deployable feature set (6–10 raw inputs).
# These are selected from the harmonized UCI-compatible schema.
DEPLOYABLE_FEATURES = [
    "studytime",    # weekly study time (1–4)
    "failures",     # past class failures (0–4)
    "absences",     # number of school absences (0–93)
    "famsup",       # family educational support (yes/no)
    "internet",     # Internet access at home (yes/no)
    "higher",       # wants to take higher education (yes/no)
    "schoolsup",    # extra educational support (yes/no)
    "health",       # current health status (1–5)
    "goout",        # going out with friends (1–5)
    "sex",          # student's sex (F/M)
]

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_uci_data():
    """Load UCI Student Performance (id=320) and return a combined DataFrame."""
    print("[load] Fetching UCI Student Performance (id=320)...")
    student_data = fetch_ucirepo(id=320)

    # Original combined dataframe is available in .data.original
    # or we can reconstruct from features + targets.
    df = student_data.data.original.copy()

    # The UCI dataset has both Math and Portuguese rows (649 total after dedup).
    # We keep both; school + subject can be inferred from G3 patterns if needed,
    # but for harmonization we just use the merged rows.
    print(f"[load] UCI shape: {df.shape}")
    return df


def load_additional_dataset_1():
    """
    Load an additional public student academic-performance dataset.
    Placeholder: replace URL/path with a real source.
    Falls back to a local CSV named 'additional_student_1.csv' if download fails.
    """
    print("[load] Attempting additional dataset 1...")
    # Example source: a Kaggle dataset harmonizable with UCI columns.
    # Replace with your actual URL.
    url = "https://example.com/additional_student_1.csv"
    local_path = "additional_student_1.csv"

    try:
        # Try local CSV first (allows manual download / offline use)
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            print(f"[load] Loaded local {local_path}: {df.shape}")
            return df
        # If no local file, attempt download
        df = pd.read_csv(url)
        print(f"[load] Downloaded additional dataset 1: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Could not load additional dataset 1. "
            f"Place '{local_path}' in the project directory or update the URL. "
            f"Original error: {e}"
        )


def load_additional_dataset_2():
    """
    Load a second additional public student academic-performance dataset.
    Same pattern: local CSV fallback first, then URL.
    """
    print("[load] Attempting additional dataset 2...")
    url = "https://example.com/additional_student_2.csv"
    local_path = "additional_student_2.csv"

    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            print(f"[load] Loaded local {local_path}: {df.shape}")
            return df
        df = pd.read_csv(url)
        print(f"[load] Downloaded additional dataset 2: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Could not load additional dataset 2. "
            f"Place '{local_path}' in the project directory or update the URL. "
            f"Original error: {e}"
        )


# ---------------------------------------------------------------------------
# 2. Dataset inspection & harmonization
# ---------------------------------------------------------------------------

def harmonize_datasets(dfs, source_names):
    """
    Align columns across datasets to a common schema.
    For UCI we map known columns. For additional datasets we attempt
    to map similar column names (case-insensitive, common aliases).
    Missing columns are added as NaN; unknown columns are dropped.
    """
    # UCI harmonized schema (subset we care about)
    common_schema = [
        "studytime", "failures", "absences", "famsup", "internet",
        "higher", "schoolsup", "health", "goout", "sex",
        "G3",  # target source
        "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "school", "paid", "activities", "nursery",
        "romantic", "famrel", "freetime", "Dalc", "Walc",
    ]

    # Column alias mapping for harmonization
    alias_map = {
        "study_time": "studytime",
        "studytime": "studytime",
        "study time": "studytime",
        "failure": "failures",
        "failures": "failures",
        "past_failures": "failures",
        "absence": "absences",
        "absences": "absences",
        "attendance": "absences",
        "family_support": "famsup",
        "famsup": "famsup",
        "internet_access": "internet",
        "internet": "internet",
        "higher_education": "higher",
        "higher": "higher",
        "school_support": "schoolsup",
        "schoolsup": "schoolsup",
        "health": "health",
        "going_out": "goout",
        "goout": "goout",
        "social": "goout",
        "gender": "sex",
        "sex": "sex",
        "final_grade": "G3",
        "g3": "G3",
        "grade": "G3",
        "age": "age",
        "address": "address",
        "family_size": "famsize",
        "famsize": "famsize",
        "parent_status": "Pstatus",
        "Pstatus": "Pstatus",
        "mother_education": "Medu",
        "Medu": "Medu",
        "father_education": "Fedu",
        "Fedu": "Fedu",
    }

    harmonized = []
    for df, name in zip(dfs, source_names):
        df = df.copy()
        # Lowercase column names for matching
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Build rename map
        rename_map = {}
        for col in df.columns:
            clean = col.replace(" ", "_").replace("-", "_")
            if clean in alias_map:
                rename_map[col] = alias_map[clean]

        df = df.rename(columns=rename_map)

        # Keep only common schema columns (add missing as NaN)
        for col in common_schema:
            if col not in df.columns:
                df[col] = np.nan

        df = df[common_schema]
        # Tag source for traceability (optional, dropped before modeling)
        df["_source"] = name
        harmonized.append(df)
        print(f"[harmonize] {name}: kept {df.shape[1]-1} common columns, {df.shape[0]} rows")

    combined = pd.concat(harmonized, ignore_index=True)
    print(f"[harmonize] Combined shape: {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------

def clean_data(df):
    """Remove duplicates, handle missing values, fix dtypes, handle invalid values."""
    print("[clean] Starting cleaning...")
    n_before = len(df)

    # --- Duplicate removal ---
    # Drop rows identical across all columns except _source (source is metadata)
    feature_cols = [c for c in df.columns if c != "_source"]
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    print(f"[clean] Duplicates removed: {n_before - len(df)}")

    # --- Invalid values ---
    # Grade columns should be 0–20
    grade_cols = [c for c in df.columns if c in ("G3",)]
    for col in grade_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 0) | (df[col] > 20), col] = np.nan

    # Absences: cap at reasonable max (93 in UCI; keep but convert)
    df["absences"] = pd.to_numeric(df["absences"], errors="coerce")
    df.loc[df["absences"] < 0, "absences"] = np.nan

    # Study time: 1–4
    df["studytime"] = pd.to_numeric(df["studytime"], errors="coerce")
    df.loc[(df["studytime"] < 1) | (df["studytime"] > 4), "studytime"] = np.nan

    # Failures: 0–4
    df["failures"] = pd.to_numeric(df["failures"], errors="coerce")
    df.loc[(df["failures"] < 0) | (df["failures"] > 4), "failures"] = np.nan

    # Health, goout: 1–5
    for col in ["health", "goout"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 1) | (df[col] > 5), col] = np.nan

    # Categorical columns: strip whitespace, lowercase
    cat_cols = ["famsup", "internet", "higher", "schoolsup", "sex",
                "address", "famsize", "Pstatus", "Mjob", "Fjob",
                "reason", "guardian", "school", "paid", "activities",
                "nursery", "romantic"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace("nan", np.nan)

    print(f"[clean] Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Target creation
# ---------------------------------------------------------------------------

def create_target(df):
    """
    Create binary target 'fail' from final grade G3.
    fail = 1 if G3 < PASS_THRESHOLD (10), else 0.
    Rows with missing G3 are dropped.
    """
    df = df.copy()
    df = df.dropna(subset=["G3"])
    df["fail"] = (df["G3"] < PASS_THRESHOLD).astype(int)

    # --- Leakage prevention ---
    # G3 is the source of the target and MUST NOT be a predictor.
    df = df.drop(columns=["G3"])

    # G1/G2 are period grades. Including them makes the prediction
    # less useful for early warning because they are only available
    # after the student has already partially failed.
    if not USE_PERIOD_GRADES:
        for col in ["G1", "G2"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        print("[target] Dropped G1/G2 to prevent leakage (USE_PERIOD_GRADES=False)")
    else:
        print("[target] Keeping G1/G2 as predictors (USE_PERIOD_GRADES=True)")

    print(f"[target] Target distribution:\n{df['fail'].value_counts()}")
    return df


# ---------------------------------------------------------------------------
# 5. Feature selection
# ---------------------------------------------------------------------------

def select_deployable_features(df, candidates=DEPLOYABLE_FEATURES):
    """
    Keep only the small deployable feature set that exists in the
    harmonized data. Fill missing categorical columns if needed.
    """
    available = [c for c in candidates if c in df.columns]
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        print(f"[features] Missing from data (will be created as NaN): {missing}")
        for col in missing:
            df[col] = np.nan

    cols = available + [c for c in missing if c not in available]
    # Ensure target is kept separately later
    keep = list(dict.fromkeys(cols))  # dedupe preserving order
    print(f"[features] Selected deployable features: {keep}")
    return df, keep


# ---------------------------------------------------------------------------
# 6. Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model, numeric_cols, categorical_cols):
    """
    Build a Pipeline with ColumnTransformer that prevents data leakage.
    Categorical: impute + one-hot encode.
    Numerical: impute + scale.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# 7. Model training & evaluation
# ---------------------------------------------------------------------------

def evaluate_model(pipeline, X_test, y_test, model_name):
    """Compute threshold- and probability-based classification metrics."""
    y_pred = pipeline.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        "PR-AUC": average_precision_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    return metrics, y_pred, y_proba


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       numeric_cols, categorical_cols):
    """Train all five models, return results, fitted pipelines, and predictions."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": None,  # placeholder; import below
        "CatBoost": None,
    }

    # Import optional libraries
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric="logloss"
        )
    except ImportError:
        print("[warn] xgboost not installed; skipping XGBoost.")

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, random_state=RANDOM_STATE, verbose=0
        )
    except ImportError:
        print("[warn] catboost not installed; skipping CatBoost.")

    results = []
    fitted = {}
    predictions = {}

    for name, model in models.items():
        if model is None:
            continue
        print(f"[train] Fitting {name}...")
        pipe = build_pipeline(model, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(pipe, X_test, y_test, name)
        results.append(metrics)
        fitted[name] = pipe
        predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}
        print(f"[train] {name}: {metrics}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    return results_df, fitted, predictions


# ---------------------------------------------------------------------------
# 8. Model selection
# ---------------------------------------------------------------------------

def print_best_models(results_df):
    """Print best by Accuracy, F1, ROC-AUC."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    for metric in ["Accuracy", "F1-Score", "ROC-AUC"]:
        best = results_df.loc[results_df[metric].idxmax()]
        print(f"\nBEST MODEL BY {metric.upper()}: {best['Model']} "
              f"({best[metric]:.4f})")


def select_deployment_model(results_df, fitted):
    """
    Select model for deployment primarily on ROC-AUC/F1/recall.
    """
    # Composite score: weight ROC-AUC and F1 and Recall
    results_df = results_df.copy()
    results_df["deploy_score"] = (
        0.4 * results_df["ROC-AUC"]
        + 0.35 * results_df["F1-Score"]
        + 0.25 * results_df["Recall"]
    )
    best_row = results_df.loc[results_df["deploy_score"].idxmax()]
    best_name = best_row["Model"]
    print(f"\n[select] Deployment model: {best_name} "
          f"(deploy_score={best_row['deploy_score']:.4f})")
    return best_name, fitted[best_name], best_row


# ---------------------------------------------------------------------------
# 9. Plots & SHAP
# ---------------------------------------------------------------------------

def plot_confusion_matrix(pipeline, X_test, y_test, model_name):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {path}")


def plot_roc_curves(predictions, y_test):
    plt.figure(figsize=(10, 6))
    for name, pred in predictions.items():
        if pred["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, pred["y_proba"])
            auc = roc_auc_score(y_test, pred["y_proba"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    path = os.path.join(ARTIFACTS_DIR, "roc_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {path}")


def plot_model_comparison(results_df):
    metrics = [
        "Accuracy", "Precision", "Recall", "Specificity", "F1-Score",
        "Balanced Accuracy", "MCC", "ROC-AUC", "PR-AUC",
    ]
    ax = results_df.set_index("Model")[metrics].plot(
        kind="bar", figsize=(16, 7), rot=45
    )
    ax.set_ylabel("Score")
    ax.set_title("Model Metric Comparison")
    ax.legend(loc="lower right")
    path = os.path.join(ARTIFACTS_DIR, "model_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {path}")

    # Save a clear, dedicated model comparison chart for every metric.
    for metric in metrics:
        metric_df = results_df.sort_values(metric, ascending=False)
        fig, metric_ax = plt.subplots(figsize=(10, 5))
        bars = metric_ax.bar(
            metric_df["Model"], metric_df[metric], color="#4C78A8"
        )
        metric_ax.set_ylabel(metric)
        metric_ax.set_title(f"{metric} Comparison Across Models")
        metric_ax.tick_params(axis="x", rotation=30)
        metric_ax.bar_label(bars, fmt="%.4f", padding=3)
        metric_ax.margins(y=0.12)
        fig.tight_layout()

        filename = "metric_" + metric.lower().replace("-", "_").replace(" ", "_") + ".png"
        metric_path = os.path.join(ARTIFACTS_DIR, filename)
        fig.savefig(metric_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved {metric_path}")


def generate_shap_explanations(pipeline, X_train, X_test, model_name):
    """
    Generate SHAP global and local explanations for the selected model.
    Uses TreeExplainer for tree-based models, LinearExplainer for linear.
    For pipelines, extracts the fitted preprocessor output and final model.
    """
    try:
        import shap
    except ImportError:
        print("[warn] shap not installed; skipping SHAP explanations.")
        return

    print(f"[shap] Generating SHAP for {model_name}...")

    # Transform test data through the pipeline's preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_test_transformed = preprocessor.transform(X_test)
    X_train_transformed = preprocessor.transform(X_train)

    # Get feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean names
        feature_names = [str(f).replace("num__", "").replace("cat__", "")
                         for f in feature_names]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_test_transformed.shape[1])]

    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Choose explainer
    model_type = type(model).__name__
    if "LogisticRegression" in model_type:
        explainer = shap.LinearExplainer(model, X_train_transformed)
    elif "DecisionTree" in model_type or "RandomForest" in model_type or \
         "XGB" in model_type or "CatBoost" in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(X_train_transformed, 100)
        )

    shap_values = explainer.shap_values(X_test_df)

    # For binary classification, take positive class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[..., 1]

    # --- Global summary plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
    path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[shap] Saved {path}")

    # --- Bar plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                      show=False, max_display=15)
    path = os.path.join(ARTIFACTS_DIR, "shap_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[shap] Saved {path}")

    # --- Individual explanation (first test sample) ---
    idx = 0
    sample_shap = shap_values[idx]
    sample_features = X_test_df.iloc[idx]

    # Top factors increasing risk (positive SHAP)
    pos_idx = np.argsort(sample_shap)[::-1]
    # Top factors decreasing risk (negative SHAP)
    neg_idx = np.argsort(sample_shap)

    increasing = [
        (feature_names[i], float(sample_shap[i]))
        for i in pos_idx if sample_shap[i] > 0
    ][:5]
    decreasing = [
        (feature_names[i], float(sample_shap[i]))
        for i in neg_idx if sample_shap[i] < 0
    ][:5]

    local_explanation = {
        "increasing_failure_risk": increasing,
        "decreasing_failure_risk": decreasing,
    }

    print(f"[shap] Top increasing risk: {increasing}")
    print(f"[shap] Top decreasing risk: {decreasing}")

    return local_explanation


def plot_feature_importance(pipeline, numeric_cols, categorical_cols):
    """Simple feature importance from the final model if available."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        print("[plot] Model has no feature_importances_ or coef_; skipping.")
        return

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        names = [str(n).replace("num__", "").replace("cat__", "") for n in names]
    except Exception:
        names = [f"f{i}" for i in range(len(importances))]

    n_show = min(15, len(importances))
    idx = np.argsort(importances)[::-1][:n_show]

    plt.figure(figsize=(10, 6))
    plt.barh(range(n_show), importances[idx][::-1])
    plt.yticks(range(n_show), [names[i] for i in idx][::-1])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {path}")


# ---------------------------------------------------------------------------
# 10. Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(pipeline, selected_features, results_df,
                   selected_model_name, local_explanation, X_train):
    """
    Save model.pkl (complete pipeline), preprocessor.pkl, feature_names.pkl,
    model_results.csv, selected_features.json, metadata.json.
    """
    print("\n[save] Saving artifacts...")

    # Complete pipeline (allows app.py to use RAW inputs directly)
    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "model.pkl"))

    # Preprocessor separately (optional for inspection)
    joblib.dump(pipeline.named_steps["preprocessor"],
                os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))

    # Feature names
    joblib.dump(selected_features,
                os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    # Results CSV
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_results.csv"),
                      index=False)

    # Selected features JSON
    with open(os.path.join(ARTIFACTS_DIR, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    # Metadata
    metadata = {
        "model_name": selected_model_name,
        "target_definition": "fail = 1 if final grade G3 < 10 else 0",
        "failure_threshold": PASS_THRESHOLD,
        "selected_features": selected_features,
        "expected_input_types": {
            "studytime": "integer (1-4)",
            "failures": "integer (0-4)",
            "absences": "integer (0-93)",
            "famsup": "categorical (yes/no)",
            "internet": "categorical (yes/no)",
            "higher": "categorical (yes/no)",
            "schoolsup": "categorical (yes/no)",
            "health": "integer (1-5)",
            "goout": "integer (1-5)",
            "sex": "categorical (F/M)",
        },
        "categorical_options": {
            "famsup": ["yes", "no"],
            "internet": ["yes", "no"],
            "higher": ["yes", "no"],
            "schoolsup": ["yes", "no"],
            "sex": ["F", "M"],
        },
        "test_metrics": results_df.set_index("Model").to_dict(orient="index"),
        "local_explanation_example": local_explanation,
        "random_state": RANDOM_STATE,
        "use_period_grades": USE_PERIOD_GRADES,
    }
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("[save] All artifacts saved to artifacts/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EXPLAINABLE STUDENT FAILURE PREDICTION — TRAINING")
    print("=" * 60)

    # 1. Load
    uci = load_uci_data()
    # Additional datasets: wrap in try/except so missing sources don't
    # block the core UCI pipeline; if they fail, we proceed with UCI only.
    additional_dfs = []
    additional_names = []
    for loader, name in [
        (load_additional_dataset_1, "additional_1"),
        (load_additional_dataset_2, "additional_2"),
    ]:
        try:
            additional_dfs.append(loader())
            additional_names.append(name)
        except Exception as e:
            print(f"[warn] Skipping {name}: {e}")

    all_dfs = [uci] + additional_dfs
    all_names = ["uci"] + additional_names

    # 2. Harmonize
    df = harmonize_datasets(all_dfs, all_names)

    # 3. Clean
    df = clean_data(df)

    # 4. Target
    df = create_target(df)

    # 5. Feature selection
    df, selected_features = select_deployable_features(df)

    # Determine numeric vs categorical from selected features
    numeric_cols = [c for c in selected_features
                    if c not in ["famsup", "internet", "higher",
                                 "schoolsup", "sex"]]
    categorical_cols = [c for c in selected_features
                        if c in ["famsup", "internet", "higher",
                                 "schoolsup", "sex"]]
    print(f"[main] Numeric: {numeric_cols}")
    print(f"[main] Categorical: {categorical_cols}")

    # Prepare X, y
    X = df[selected_features].copy()
    y = df["fail"].copy()

    # Drop rows where target is missing (already handled but double-check)
    valid = y.notna()
    X, y = X[valid], y[valid]

    # 6. Class imbalance inspection
    print(f"[main] Class distribution:\n{y.value_counts(normalize=True)}")

    # 7. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[main] Train: {X_train.shape}, Test: {X_test.shape}")

    # 8. Train models
    results_df, fitted_pipelines, predictions = train_and_evaluate(
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
    )

    # 9. Selection
    print_best_models(results_df)
    selected_name, selected_pipeline, selected_row = select_deployment_model(
        results_df, fitted_pipelines
    )

    # 10. Plots & SHAP
    plot_confusion_matrix(selected_pipeline, X_test, y_test, selected_name)
    plot_roc_curves(predictions, y_test)
    plot_model_comparison(results_df)
    plot_feature_importance(selected_pipeline, numeric_cols, categorical_cols)

    local_explanation = generate_shap_explanations(
        selected_pipeline, X_train, X_test, selected_name
    )

    # 11. Save
    save_artifacts(
        selected_pipeline, selected_features, results_df,
        selected_name, local_explanation, X_train
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
