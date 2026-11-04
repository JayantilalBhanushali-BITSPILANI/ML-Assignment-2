"""
app.py — Credit Card Default Risk: Classifier Evaluation Console
================================================================
Streamlit front-end for ML Assignment 2 (BITS Pilani WILP, M.Tech AIML).

The app NEVER retrains. It loads the 5 pre-trained pipelines from ./model,
applies the shared feature engineering (src/features.py) to the uploaded
test CSV, and reports metrics, confusion matrix, classification report and
ROC curve.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features import RAW_FEATURES, TARGET_COL, prepare_features, validate_columns

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST_CSV = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
MODEL_BLURB = {
    "Logistic Regression": "Linear model on standardized features; outputs calibrated-ish probabilities and is easy to explain to risk teams.",
    "Decision Tree": "Rule-based splits (entropy, depth-limited) — captures non-linear thresholds such as 'delayed 2+ months'.",
    "KNN": "Classifies a customer by majority vote of the most similar customers in the training set (scaled features).",
    "Naive Bayes": "Gaussian NB on Yeo-Johnson transformed features; assumes features are conditionally independent.",
    "Support Vector Machine": "RBF-kernel SVM; finds a maximum-margin non-linear boundary. AUC uses the decision-function score.",
}
CLASS_NAMES = ["No Default (0)", "Default (1)"]
PRIMARY = "#1f4e79"

st.set_page_config(page_title="Credit Default Risk – Model Console", page_icon="💳", layout="wide")

# Light custom styling (keeps the UI distinct from the stock template)
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.6rem;}}
      .hero {{background: linear-gradient(90deg, {PRIMARY} 0%, #2e75b6 100%);
              padding: 1.1rem 1.4rem; border-radius: 12px; color: white; margin-bottom: 1rem;}}
      .hero h1 {{color: white; font-size: 1.7rem; margin: 0;}}
      .hero p  {{color: #dbe8f5; margin: .25rem 0 0 0; font-size: .95rem;}}
      div[data-testid="stMetric"] {{background: #f4f8fc; border: 1px solid #d6e4f0;
              border-radius: 10px; padding: .6rem .8rem;}}
    </style>
    <div class="hero">
      <h1>💳 Credit Card Default Risk — Classifier Evaluation Console</h1>
      <p>Evaluate 5 pre-trained classifiers on held-out test data · UCI "Default of Credit Card Clients" ·
         ML Assignment 2, BITS Pilani WILP (M.Tech AIML)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained models…")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_training_metrics() -> pd.DataFrame | None:
    p = MODEL_DIR / "metrics_summary.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    p = MODEL_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def scores_for_auc(model, X) -> np.ndarray:
    """Positive-class probability, or SVM decision function when proba is unavailable."""
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:
        return model.decision_function(X)


def evaluate_model(model, X: pd.DataFrame, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    y_score = scores_for_auc(model, X)
    auc = roc_auc_score(y, y_score) if len(np.unique(y)) == 2 else np.nan
    return {
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": auc,
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
        },
    }


def plot_confusion(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    pct = cm / cm.sum() * 100
    labels = np.array([[f"{v:,}\n({p:.1f}%)" for v, p in zip(r1, r2)] for r1, r2 in zip(cm, pct)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 11})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc(curves: dict):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC curve", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar – data source + model selection
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1️⃣  Test data")
    source = st.radio("Choose data source", ["Upload a CSV", "Use bundled test_data.csv"], index=0)
    uploaded = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Upload test CSV (features + target)", type=["csv"])
    if SAMPLE_TEST_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_TEST_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch")

    st.header("2️⃣  Model")
    model_name = st.selectbox("Select a classifier", list(MODEL_FILES.keys()), index=0)
    st.caption(MODEL_BLURB[model_name])

    with st.expander("Required CSV columns"):
        st.code(", ".join(RAW_FEATURES + [TARGET_COL]), language="text")
        st.caption("Column order does not matter. Extra columns (e.g. ID) are ignored.")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
if source == "Upload a CSV" and uploaded is None:
    st.info("👈 Upload a test CSV from the sidebar, or switch to **Use bundled test_data.csv** to try the app instantly.")
    train_metrics = load_training_metrics()
    if train_metrics is not None:
        st.subheader("Reference: test-set metrics recorded at training time")
        st.dataframe(train_metrics, width="stretch")
    st.stop()

try:
    df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(SAMPLE_TEST_CSV)
except Exception as exc:  # noqa: BLE001 – show any parsing error to the user
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

missing = validate_columns(df, require_target=True)
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s): `{', '.join(missing)}`")
    st.stop()

df = df.dropna(subset=RAW_FEATURES + [TARGET_COL])
if df.empty:
    st.error("No usable rows after removing rows with missing values.")
    st.stop()

y_true = df[TARGET_COL].astype(int).to_numpy()
X = prepare_features(df)
models = load_models()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_eval, tab_compare, tab_data, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔍 Data preview", "ℹ️ About"]
)

# ---- Tab 1: selected model ------------------------------------------------ #
with tab_eval:
    res = evaluate_model(models[model_name], X, y_true)
    m = res["metrics"]

    st.subheader(f"{model_name} — results on {len(df):,} test records")
    cols = st.columns(6)
    for col, key in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        col.metric(key, "n/a" if np.isnan(m[key]) else f"{m[key]:.4f}")

    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_true, res["y_pred"], labels=[0, 1])
        st.pyplot(plot_confusion(cm, f"Confusion matrix — {model_name}"), width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Correctly flagged defaulters (TP): **{tp:,}** · Missed defaulters (FN): **{fn:,}** · "
                   f"False alarms (FP): **{fp:,}**")
    with right:
        if not np.isnan(m["AUC"]):
            fpr, tpr, _ = roc_curve(y_true, res["y_score"])
            st.pyplot(plot_roc({model_name: (fpr, tpr, m["AUC"])}), width="stretch")

    st.markdown("##### Classification report")
    report = classification_report(y_true, res["y_pred"], labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    out = df.copy()
    out["predicted_default"] = res["y_pred"]
    out["model_score"] = np.round(res["y_score"], 5)
    st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv")

# ---- Tab 2: all models side-by-side -------------------------------------- #
with tab_compare:
    st.subheader("All 5 models on the same test data")
    all_res = {name: evaluate_model(mdl, X, y_true) for name, mdl in models.items()}
    table = pd.DataFrame({n: r["metrics"] for n, r in all_res.items()}).T.round(4)
    table.index.name = "ML Model Name"
    st.dataframe(table, width="stretch")

    best = {k: table[k].idxmax() for k in ["Accuracy", "AUC", "F1", "MCC"]}
    st.success("  ·  ".join(f"Best **{k}**: {v}" for k, v in best.items()))

    c1, c2 = st.columns(2)
    with c1:
        long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", ax=ax, palette="Blues_d")
        ax.set_ylim(0, 1)
        ax.set_title("Metric comparison", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with c2:
        if len(np.unique(y_true)) == 2:
            curves = {}
            for n, r in all_res.items():
                fpr, tpr, _ = roc_curve(y_true, r["y_score"])
                curves[n] = (fpr, tpr, r["metrics"]["AUC"])
            st.pyplot(plot_roc(curves), width="stretch")

    st.download_button("⬇️ Download comparison table (CSV)", table.to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# ---- Tab 3: data preview -------------------------------------------------- #
with tab_data:
    st.subheader("Uploaded data")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Input features (raw)", len(RAW_FEATURES))
    c.metric("Default rate", f"{y_true.mean():.2%}")
    st.dataframe(df.head(50), width="stretch")
    with st.expander("Engineered features fed to the models (first 20 rows)"):
        st.dataframe(X.head(20), width="stretch")

# ---- Tab 4: about --------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
**Problem.** Predict whether a credit-card customer will **default on next month's payment**
(binary: 1 = default, 0 = no default) from demographic data, credit limit, 6 months of
repayment status, bill amounts and payment amounts.

**Pipeline.** Raw CSV → category clean-up (EDUCATION / MARRIAGE) → 5 engineered risk features
(utilization, payment ratio, months delayed, max delay) → model-specific preprocessing
(one-hot + scaling / power transform) → classifier. All preprocessing is stored *inside*
each saved pipeline, so the app only calls `predict`.

**No retraining happens here** — models are loaded from `model/*.joblib`.
        """
    )
    meta, params = load_json("metadata.json"), load_json("best_params.json")
    if meta:
        st.markdown("##### Training metadata")
        st.json(meta, expanded=False)
    if params:
        st.markdown("##### Tuned hyper-parameters (GridSearchCV, scoring = ROC-AUC)")
        st.json(params, expanded=False)
