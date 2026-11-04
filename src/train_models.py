"""
train_models.py
---------------
End-to-end training script for ML Assignment 2 (BITS Pilani WILP, M.Tech AIML).

What it does
1. Loads the UCI "Default of Credit Card Clients" dataset (data/UCI_Credit_Card.csv)
2. Removes the ID column and exact duplicate records
3. Stratified 80/20 train-test split
     -> test split is saved as ./test_data.csv (raw columns + target),
        which is the file you upload to the Streamlit app
4. Builds 5 sklearn Pipelines (preprocessing + classifier):
     Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM (RBF)
5. Light hyper-parameter tuning with Stratified K-Fold CV (scoring = ROC-AUC)
6. Evaluates every model on the held-out test set with 6 metrics:
     Accuracy, AUC, Precision, Recall, F1, MCC
7. Saves:  model/<model>.joblib, model/metrics_summary.csv,
           model/best_params.json, model/metadata.json

Run from the project root:
    python -m src.train_models
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.features import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    TARGET_COL,
    prepare_features,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "UCI_Credit_Card.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV_PATH = ROOT / "test_data.csv"

RANDOM_STATE = 2026      # change this to get a different (but reproducible) split
TEST_SIZE = 0.20
CV_FOLDS = 5
SVM_TUNING_SAMPLE = 8000  # SVM is O(n^2); tune on a stratified subsample, refit on full train

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# --------------------------------------------------------------------------- #
# Pre-processing blocks
# --------------------------------------------------------------------------- #
def make_preprocessor(numeric_scaler) -> ColumnTransformer:
    """One-hot encode nominal categoricals; scale/transform numeric columns."""
    num = numeric_scaler if numeric_scaler is not None else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", num, NUMERIC_FEATURES),
        ]
    )


def build_model_specs() -> dict:
    """Pipeline + hyper-parameter grid for each of the 5 required models."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("prep", make_preprocessor(StandardScaler())),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            # Trees are scale-invariant -> no scaling needed
            Pipeline([
                ("prep", make_preprocessor(None)),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "clf__max_depth": [3, 4, 5, 6, 8, 10],
                "clf__min_samples_leaf": [20, 50, 100],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "KNN": (
            # Distance-based -> scaling is mandatory
            Pipeline([
                ("prep", make_preprocessor(StandardScaler())),
                ("clf", KNeighborsClassifier()),
            ]),
            {"clf__n_neighbors": [15, 31, 51, 75], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            # GaussianNB assumes normally distributed features; the monetary columns are
            # heavily right-skewed, so a Yeo-Johnson PowerTransformer is applied first.
            Pipeline([
                ("prep", make_preprocessor(PowerTransformer(method="yeo-johnson"))),
                ("clf", GaussianNB()),
            ]),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
        ),
        "Support Vector Machine": (
            Pipeline([
                ("prep", make_preprocessor(StandardScaler())),
                ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.3, 1.0, 3.0], "clf__gamma": ["scale", 0.01]},
        ),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def get_scores(model, X) -> np.ndarray:
    """Continuous score for AUC: probability if available, else SVM decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # SVC without probability=True
            pass
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load -------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"[1] Loaded {DATA_PATH.name}: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # 2. Basic cleaning ---------------------------------------------------
    df = df.drop(columns=[ID_COL])
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[2] Dropped ID column and {n_before - len(df)} duplicate rows -> {len(df):,} rows")
    print(f"    Class balance: {df[TARGET_COL].value_counts(normalize=True).round(4).to_dict()}")

    # 3. Split (raw columns kept so test_data.csv is in the original UCI format) ---
    train_raw, test_raw = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    test_raw[RAW_FEATURES + [TARGET_COL]].to_csv(TEST_CSV_PATH, index=False)
    print(f"[3] Train: {len(train_raw):,} | Test: {len(test_raw):,}  -> saved {TEST_CSV_PATH.name}")

    X_train, y_train = prepare_features(train_raw), train_raw[TARGET_COL].values
    X_test, y_test = prepare_features(test_raw), test_raw[TARGET_COL].values
    print(f"    Model feature count: {X_train.shape[1]} (23 raw + 5 engineered)")

    # 4-6. Train, tune, evaluate -------------------------------------------
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, best_params = [], {}

    for name, (pipe, grid) in build_model_specs().items():
        t = time.time()
        if name == "Support Vector Machine":
            # Tune on a stratified subsample (SVM training cost grows ~quadratically)
            X_tune, _, y_tune, _ = train_test_split(
                X_train, y_train, train_size=SVM_TUNING_SAMPLE,
                stratify=y_train, random_state=RANDOM_STATE,
            )
            search = GridSearchCV(pipe, grid, scoring="roc_auc",
                                  cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
            search.fit(X_tune, y_tune)
            model = pipe.set_params(**search.best_params_).fit(X_train, y_train)
        else:
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_

        metrics = evaluate(model, X_test, y_test)
        results.append({"ML Model Name": name, **metrics})
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)

        print(f"[4] {name:<24} CV-AUC={search.best_score_:.4f}  "
              f"Test-AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  ({time.time() - t:.1f}s)")

    # 7. Persist results ----------------------------------------------------
    summary = pd.DataFrame(results).set_index("ML Model Name").round(4)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv")
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "dataset": "UCI Default of Credit Card Clients (id=350)",
        "trained_on": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(train_raw)),
        "n_test": int(len(test_raw)),
        "n_model_features": int(X_train.shape[1]),
        "sklearn_version": sklearn.__version__,
        "model_files": MODEL_FILES,
    }, indent=2))

    print("\n================  TEST-SET COMPARISON  ================")
    print(summary.to_string())
    print(f"\nBest model by MCC: {summary['MCC'].idxmax()} | by AUC: {summary['AUC'].idxmax()}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
