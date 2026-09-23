"""
app.py — Explainable Student Failure Prediction System (Streamlit dashboard)

Run with:
    streamlit run app.py

Loads the trained pipeline and artifacts produced by train.py.
Does NOT retrain models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "model_results.csv")
SELECTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "selected_features.json")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

COMPARISON_METRICS = [
    "Accuracy", "Precision", "Recall", "Specificity", "F1-Score",
    "Balanced Accuracy", "MCC", "ROC-AUC", "PR-AUC",
]

st.set_page_config(
    page_title="Student Academic Risk Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Readable feature labels
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "studytime": "Weekly Study Time",
    "failures": "Previous Class Failures",
    "absences": "School Absences",
    "famsup": "Family Educational Support",
    "internet": "Internet Access at Home",
    "higher": "Intends Higher Education",
    "schoolsup": "Extra School Support",
    "health": "Current Health Status",
    "goout": "Going Out with Friends",
    "sex": "Student Sex",
    # extended / harmonized columns
    "age": "Age",
    "address": "Home Address Type",
    "famsize": "Family Size",
    "Pstatus": "Parent Cohabitation Status",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "Mjob": "Mother's Job",
    "Fjob": "Father's Job",
    "reason": "Reason for Choosing School",
    "guardian": "Guardian",
    "traveltime": "Travel Time to School",
    "school": "School",
    "paid": "Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery",
    "romantic": "In a Romantic Relationship",
    "famrel": "Family Relationship Quality",
    "freetime": "Free Time After School",
    "Dalc": "Workday Alcohol Consumption",
    "Walc": "Weekend Alcohol Consumption",
    "G1": "First Period Grade",
    "G2": "Second Period Grade",
}

# Friendly widget configs per feature
NUMERIC_WIDGETS = {
    "studytime": {"min": 1, "max": 4, "default": 2,
                  "help": "1 = <2h, 2 = 2–5h, 3 = 5–10h, 4 = >10h per week"},
    "failures": {"min": 0, "max": 4, "default": 0,
                 "help": "Number of past class failures (0–4)"},
    "absences": {"min": 0, "max": 93, "default": 4,
                 "help": "Number of school absences"},
    "health": {"min": 1, "max": 5, "default": 3,
               "help": "1 = very bad … 5 = very good"},
    "goout": {"min": 1, "max": 5, "default": 3,
              "help": "1 = very low … 5 = very high social activity"},
    "age": {"min": 15, "max": 22, "default": 17, "help": "Student age"},
    "Medu": {"min": 0, "max": 4, "default": 2,
             "help": "0 = none … 4 = higher education"},
    "Fedu": {"min": 0, "max": 4, "default": 2,
             "help": "0 = none … 4 = higher education"},
    "traveltime": {"min": 1, "max": 4, "default": 1,
                   "help": "1 = <15 min … 4 = >1 hour"},
    "famrel": {"min": 1, "max": 5, "default": 4,
               "help": "1 = very bad … 5 = excellent"},
    "freetime": {"min": 1, "max": 5, "default": 3,
                 "help": "1 = very low … 5 = very high"},
    "Dalc": {"min": 1, "max": 5, "default": 1,
             "help": "1 = very low … 5 = very high"},
    "Walc": {"min": 1, "max": 5, "default": 2,
             "help": "1 = very low … 5 = very high"},
}

CATEGORICAL_OPTIONS = {
    "famsup": ["yes", "no"],
    "internet": ["yes", "no"],
    "higher": ["yes", "no"],
    "schoolsup": ["yes", "no"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "school": ["GP", "MS"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "romantic": ["yes", "no"],
}

CATEGORICAL_FEATURES = set(CATEGORICAL_OPTIONS.keys())


# ---------------------------------------------------------------------------
# Artifact loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_selected_features():
    if not os.path.exists(SELECTED_FEATURES_PATH):
        return None
    with open(SELECTED_FEATURES_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)


def artifacts_ready():
    """Check that the minimum required artifacts exist."""
    return all(os.path.exists(p) for p in [MODEL_PATH, SELECTED_FEATURES_PATH,
                                           METADATA_PATH, RESULTS_PATH])


def missing_artifacts_message():
    required = {
        "model.pkl": MODEL_PATH,
        "selected_features.json": SELECTED_FEATURES_PATH,
        "metadata.json": METADATA_PATH,
        "model_results.csv": RESULTS_PATH,
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    st.error(
        "Training artifacts are missing. Please run `python train.py` first to "
        "generate them in the `artifacts/` directory."
    )
    if missing:
        st.info("Missing files: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def feature_label(name):
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def build_input_row(feature_list, values):
    """Construct a one-row DataFrame with raw inputs in the expected column order."""
    return pd.DataFrame([{f: values[f] for f in feature_list}])


def extract_shap_for_instance(pipeline, raw_row):
    """
    Compute SHAP values for a single raw input row.
    Returns (shap_values_for_positive_class, feature_names_after_transform).
    """
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(raw_row)

    try:
        names = preprocessor.get_feature_names_out()
        names = [str(n).replace("num__", "").replace("cat__", "") for n in names]
    except Exception:
        names = [f"f{i}" for i in range(X_transformed.shape[1])]

    model_type = type(model).__name__
    try:
        if "LogisticRegression" in model_type:
            explainer = shap.LinearExplainer(model, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
        elif ("DecisionTree" in model_type or "RandomForest" in model_type
              or "XGB" in model_type or "CatBoost" in model_type):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        raise RuntimeError(f"SHAP explanation failed: {e}")

    # Handle binary-class output shapes
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[..., 1]

    return np.asarray(shap_values).ravel(), list(names)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Predict Student Risk", "Explain Prediction",
     "Model Performance", "About"],
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Student Academic Risk Prediction Dashboard")
st.caption(
    "Early identification of students who may require academic support "
    "using Machine Learning and Explainable AI."
)
st.divider()


# ---------------------------------------------------------------------------
# Guard: artifacts
# ---------------------------------------------------------------------------
if not artifacts_ready():
    missing_artifacts_message()
    st.stop()

model = load_model()
metadata = load_metadata()
selected_features = load_selected_features()
results_df = load_results()

if model is None or metadata is None or selected_features is None:
    missing_artifacts_message()
    st.stop()


# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview")

    deployed_model = metadata.get("model_name", "Unknown")
    test_metrics = metadata.get("test_metrics", {})
    deployed_metrics = test_metrics.get(deployed_model, {})
    n_features = len(selected_features)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Deployed Model", deployed_model)
    c2.metric("Accuracy", f"{deployed_metrics.get('Accuracy', float('nan')):.3f}")
    c3.metric("F1-Score", f"{deployed_metrics.get('F1-Score', float('nan')):.3f}")
    c4.metric("ROC-AUC", f"{deployed_metrics.get('ROC-AUC', float('nan')):.3f}")
    c5.metric("Selected Features", n_features)

    st.divider()
    st.markdown("### Model Comparison")
    if results_df is not None:
        st.dataframe(
            results_df.style.format({
                metric: "{:.4f}" for metric in COMPARISON_METRICS
                if metric in results_df.columns
            }).highlight_max(
                subset=[m for m in COMPARISON_METRICS if m in results_df.columns],
                color="#d4f4dd",
            ),
            use_container_width=True,
        )
    else:
        st.info("model_results.csv not found.")

    st.markdown("### Target Definition")
    st.write(metadata.get("target_definition", "N/A"))
    st.caption(
        f"Failure threshold: final grade < {metadata.get('failure_threshold', 'N/A')} "
        f"(on a 0–20 scale)."
    )


# ---------------------------------------------------------------------------
# PREDICT STUDENT RISK
# ---------------------------------------------------------------------------
elif page == "Predict Student Risk":
    st.subheader("Predict Student Risk")
    st.write(
        "Fill in the student profile below. Inputs are passed directly to the "
        "saved pipeline (raw values are used — no manual encoding needed)."
    )

    # Initialize session state for inputs and prediction
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "student_values" not in st.session_state:
        st.session_state.student_values = None

    # Dynamically create widgets from selected features
    inputs = {}
    with st.form("student_form"):
        col_a, col_b = st.columns(2)

        numeric_feats = [f for f in selected_features
                         if f not in CATEGORICAL_FEATURES]
        categorical_feats = [f for f in selected_features
                             if f in CATEGORICAL_FEATURES]

        half = (len(numeric_feats) + 1) // 2
        numeric_left = numeric_feats[:half]
        numeric_right = numeric_feats[half:]

        with col_a:
            for feat in numeric_left:
                cfg = NUMERIC_WIDGETS.get(feat, {"min": 0, "max": 100,
                                                 "default": 0, "help": ""})
                inputs[feat] = st.slider(
                    feature_label(feat),
                    min_value=int(cfg["min"]),
                    max_value=int(cfg["max"]),
                    value=int(cfg["default"]),
                    help=cfg.get("help", ""),
                )
            for feat in categorical_feats[: (len(categorical_feats) + 1) // 2]:
                opts = CATEGORICAL_OPTIONS.get(feat, ["yes", "no"])
                inputs[feat] = st.selectbox(
                    feature_label(feat), options=opts,
                )

        with col_b:
            for feat in numeric_right:
                cfg = NUMERIC_WIDGETS.get(feat, {"min": 0, "max": 100,
                                                 "default": 0, "help": ""})
                inputs[feat] = st.slider(
                    feature_label(feat),
                    min_value=int(cfg["min"]),
                    max_value=int(cfg["max"]),
                    value=int(cfg["default"]),
                    help=cfg.get("help", ""),
                )
            for feat in categorical_feats[(len(categorical_feats) + 1) // 2:]:
                opts = CATEGORICAL_OPTIONS.get(feat, ["yes", "no"])
                inputs[feat] = st.selectbox(
                    feature_label(feat), options=opts,
                )

        submitted = st.form_submit_button("Predict Risk", type="primary")

    if submitted:
        try:
            raw_row = build_input_row(selected_features, inputs)
            proba = float(model.predict_proba(raw_row)[0, 1])
            pred_class = int(model.predict(raw_row)[0])

            st.session_state.prediction = {
                "proba": proba,
                "class": pred_class,
                "row": raw_row,
            }
            st.session_state.student_values = inputs
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Display prediction if available
    if st.session_state.prediction is not None:
        p = st.session_state.prediction
        proba = p["proba"]
        pred_class = p["class"]

        st.divider()
        st.markdown("### Estimated Academic Risk")

        c1, c2 = st.columns([1, 2])
        with c1:
            if pred_class == 1:
                st.error("⚠️ HIGH RISK")
            else:
                st.success("✅ LOW RISK")
        with c2:
            st.metric("Failure Probability", f"{proba * 100:.1f}%")
            st.progress(min(max(proba, 0.0), 1.0))

        st.caption(
            "This is an estimated academic risk score from a machine learning "
            "model. It is not a guaranteed outcome. Use it only as a support "
            "signal alongside human judgement."
        )


# ---------------------------------------------------------------------------
# EXPLAIN PREDICTION
# ---------------------------------------------------------------------------
elif page == "Explain Prediction":
    st.subheader("Explain Prediction")
    st.markdown("#### Why did the model make this prediction?")

    if st.session_state.get("prediction") is None:
        st.info(
            "Please make a prediction first on the **Predict Student Risk** page. "
            "The explanation will then appear here."
        )
        st.stop()

    p = st.session_state.prediction
    raw_row = p["row"]
    proba = p["proba"]

    st.markdown(
        f"**Current prediction:** "
        f"{'HIGH RISK' if p['class'] == 1 else 'LOW RISK'} — "
        f"estimated failure probability **{proba * 100:.1f}%**."
    )

    try:
        shap_vals, transformed_names = extract_shap_for_instance(model, raw_row)
    except Exception as e:
        st.warning(f"SHAP explanation could not be computed: {e}")
        st.stop()

    # Build a DataFrame of transformed feature → shap value
    contributions = pd.DataFrame({
        "feature": transformed_names,
        "shap": shap_vals,
    })

    # Aggregate one-hot encoded categorical features back into a single
    # readable feature by summing their SHAP contributions.
    def aggregate_feature(name):
        # e.g. "sex_F" -> "sex"; "famsup_yes" -> "famsup"
        for feat in selected_features:
            if name == feat or name.startswith(feat + "_"):
                return feat
        return name

    contributions["group"] = contributions["feature"].apply(aggregate_feature)
    grouped = (
        contributions.groupby("group", as_index=False)["shap"].sum()
        .sort_values("shap", ascending=False)
    )

    # Map to readable labels
    grouped["label"] = grouped["group"].apply(feature_label)

    increasing = grouped[grouped["shap"] > 0].sort_values("shap", ascending=False).head(5)
    decreasing = grouped[grouped["shap"] < 0].sort_values("shap", ascending=True).head(5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔺 Factors increasing predicted risk")
        if len(increasing) == 0:
            st.write("None — no features pushed risk upward.")
        else:
            for _, r in increasing.iterrows():
                st.write(f"- **{r['label']}** (+{r['shap']:.3f})")

    with c2:
        st.markdown("##### 🔻 Factors decreasing predicted risk")
        if len(decreasing) == 0:
            st.write("None — no features pushed risk downward.")
        else:
            for _, r in decreasing.iterrows():
                st.write(f"- **{r['label']}** ({r['shap']:.3f})")

    # SHAP bar chart
    st.markdown("##### SHAP Contribution Chart")
    top_n = 10
    top = grouped.reindex(
        grouped["shap"].abs().sort_values(ascending=False).index
    ).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(top))))
    colors = ["#d64545" if v > 0 else "#3d9970" for v in top["shap"]]
    ax.barh(top["label"], top["shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on failure probability)")
    ax.set_title("Top factors influencing this prediction")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Positive SHAP values push the model toward predicting failure; "
        "negative values push toward passing."
    )


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance")

    if results_df is None:
        st.info("model_results.csv not found.")
        st.stop()

    fmt_df = results_df.copy()
    for col in COMPARISON_METRICS:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].astype(float).round(4)

    st.markdown("### Comparison Table")
    st.dataframe(
        fmt_df.style.highlight_max(
            subset=[c for c in COMPARISON_METRICS if c in fmt_df.columns],
            color="#d4f4dd",
        ),
        use_container_width=True,
    )

    # Best-per-metric callouts
    c1, c2, c3 = st.columns(3)
    for col, name, container in [
        ("Accuracy", "Best Accuracy", c1),
        ("F1-Score", "Best F1-Score", c2),
        ("ROC-AUC", "Best ROC-AUC", c3),
    ]:
        if col in results_df.columns:
            best = results_df.loc[results_df[col].idxmax()]
            container.metric(name, best["Model"], f"{best[col]:.4f}")

    st.divider()
    st.markdown("### Plots from artifacts/")

    plot_files = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("roc_curves.png", "ROC Curves"),
        ("model_comparison.png", "Model Metric Comparison"),
        ("feature_importance.png", "Feature Importance"),
        ("shap_summary.png", "SHAP Summary"),
        ("shap_bar.png", "SHAP Bar"),
    ]

    any_plot = False
    for fname, caption in plot_files:
        path = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(path):
            any_plot = True
            st.markdown(f"**{caption}**")
            st.image(path, use_container_width=True)

    if not any_plot:
        st.info("No plots found in artifacts/. Run train.py to generate them.")

    st.divider()
    st.markdown("### Individual Metric Comparisons")
    metric_columns = st.columns(2)
    for index, metric in enumerate(COMPARISON_METRICS):
        filename = "metric_" + metric.lower().replace("-", "_").replace(" ", "_") + ".png"
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            with metric_columns[index % 2]:
                st.markdown(f"**{metric}**")
                st.image(path, use_container_width=True)


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About":
    st.subheader("About This Project")

    st.markdown("### Objective")
    st.write(
        "Predict whether a student is at risk of academic failure and explain "
        "the prediction using Explainable AI (SHAP). The goal is to provide "
        "an early-warning support signal for educators and advisors."
    )

    st.markdown("### Datasets")
    st.write(
        "- **UCI Student Performance** (id=320) — primary source, includes "
        "Math and Portuguese language course records with grades G1, G2, G3.\n"
        "- **Additional student academic-performance datasets** — harmonized "
        "to a common schema where possible. Local CSV fallbacks are supported."
    )

    st.markdown("### Preprocessing")
    st.write(
        "- Duplicate removal, missing-value imputation, invalid-value correction.\n"
        "- Categorical encoding via one-hot (inside the pipeline).\n"
        "- Numerical scaling via StandardScaler.\n"
        "- All preprocessing is fit on the training split only — no data leakage."
    )

    st.markdown("### Target Definition")
    st.write(metadata.get("target_definition", "N/A"))
    st.caption(
        f"Failure threshold: final grade < {metadata.get('failure_threshold', 'N/A')} "
        f"(0–20 scale). Final grade (G3) is excluded from predictors to prevent leakage."
    )

    st.markdown("### Machine Learning Models")
    st.write(
        "Logistic Regression, Decision Tree, Random Forest, XGBoost, and CatBoost "
        "were trained and compared on Accuracy, Precision, Recall, Specificity, "
        "F1-Score, Balanced Accuracy, MCC, ROC-AUC, and PR-AUC."
    )

    st.markdown("### Model Evaluation")
    st.write(
        "Stratified train/test split. ROC-AUC is computed from predicted "
        "probabilities. Best model is selected using a weighted combination of "
        "ROC-AUC, F1-Score, and Recall — prioritizing identification of at-risk students."
    )

    st.markdown("### Explainable AI (SHAP)")
    st.write(
        "Global and local SHAP explanations highlight which features push a "
        "student's predicted risk upward or downward. Categorical one-hot "
        "contributions are aggregated back into readable feature labels."
    )

    st.markdown("### Limitations")
    st.write(
        "- Dataset is limited in size and scope; it may not generalize to all "
        "schools or student populations.\n"
        "- Predictions are based on a small set of features and cannot capture "
        "the full complexity of a student's circumstances.\n"
        "- Correlation is not causation — SHAP values indicate model behavior, "
        "not causal effects."
    )

    st.warning(
        "This system is an educational decision-support prototype and should not "
        "be used as the sole basis for decisions affecting students."
    )
