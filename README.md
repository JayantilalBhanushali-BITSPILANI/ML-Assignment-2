# Credit Card Default Prediction — Comparison of 5 Classification Models

**ML Assignment 2 · M.Tech (AIML) · BITS Pilani WILP**

> 🔗 **Live Streamlit App:** https://jayantilal-ml-assignment-2.streamlit.app/
>
> 🔗 **GitHub Repository:** https://github.com/JayantilalBhanushali-BITSPILANI/ML-Assignment-2

---

## a. Problem Statement

Credit-card issuers lose money when customers fail to pay their dues. The goal of this project is to
**predict whether a credit-card customer will default on next month's payment** (binary classification:
`1 = default`, `0 = no default`) using the customer's credit limit, demographics, and their last six months of
repayment status, bill amounts and payment amounts.

Five classification algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes
and Support Vector Machine — are trained on the **same** dataset and the **same** train/test split, evaluated
on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and deployed in an interactive Streamlit app that
scores an uploaded test CSV using the saved models (no retraining in the app).

---

## b. Dataset Description

| Item | Details |
|---|---|
| **Name** | Default of Credit Card Clients |
| **Source** | UCI Machine Learning Repository (ID 350) — <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients> (also on Kaggle: `uciml/default-of-credit-card-clients-dataset`) |
| **Citation** | Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480. |
| **Instances** | 30,000 customers (Taiwan, Apr–Sep 2005) → 29,965 after removing 35 exact duplicates |
| **Raw input features** | 23 (requirement: ≥ 12) ✅ |
| **Features used by models** | 28 = 23 raw + 5 engineered |
| **Target** | `default.payment.next.month` (1 = default, 0 = no default) |
| **Class balance** | 77.9 % no-default / 22.1 % default (imbalanced) |
| **Missing values** | None |

**Feature groups**

| Group | Columns | Meaning |
|---|---|---|
| Credit limit | `LIMIT_BAL` | Credit given (NT$), incl. supplementary cards |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | 1=male/2=female; 1=grad school, 2=university, 3=high school, 4=others; 1=married, 2=single, 3=others |
| Repayment status | `PAY_0`, `PAY_2` … `PAY_6` | Sep→Apr 2005; −2/−1/0 = paid on time / no use, 1…9 = months of delay |
| Bill amounts | `BILL_AMT1` … `BILL_AMT6` | Monthly statement amount (NT$), Sep→Apr |
| Payment amounts | `PAY_AMT1` … `PAY_AMT6` | Amount actually paid (NT$), Sep→Apr |

**Pre-processing & feature engineering** (`src/features.py`, shared by training and the app)

1. Dropped `ID`; removed 35 exact duplicate rows (to avoid train/test leakage).
2. Undocumented category codes merged: `EDUCATION ∈ {0,5,6} → 4 (others)`, `MARRIAGE = 0 → 3 (others)`.
3. Five engineered credit-risk features:
   - `AVG_UTILIZATION` = mean bill amount / credit limit
   - `LATEST_UTILIZATION` = latest bill / credit limit
   - `PAYMENT_RATIO` = total paid / total billed (clipped to [0, 2])
   - `NUM_MONTHS_DELAYED` = number of months with a payment delay
   - `MAX_DELAY` = worst delay in the last 6 months
4. Model-specific preprocessing inside each saved pipeline: one-hot encoding of `SEX/EDUCATION/MARRIAGE`;
   `StandardScaler` for LR/KNN/SVM; `PowerTransformer` (Yeo-Johnson) for Naive Bayes; no scaling for the tree.
5. **Stratified 80/20 split** (`random_state = 2026`) → 23,972 train / 5,993 test rows.
   The test split is saved as `test_data.csv` in raw UCI format.

---

## c. GitHub Repository Link

https://github.com/JayantilalBhanushali-BITSPILANI/ML-Assignment-2

**Repository structure**

```
ml-assignment-2/
│-- app.py                     # Streamlit app (loads saved models, no retraining)
│-- requirements.txt           # Pinned dependencies
│-- README.md                  # This file
│-- test_data.csv              # Held-out test set (5,993 rows, 23 features + target)
│-- data/
│   └── UCI_Credit_Card.csv    # Full raw dataset (30,000 rows)
│-- src/
│   ├── features.py            # Cleaning + feature engineering (shared)
│   └── train_models.py        # Training, tuning, evaluation, model saving
│-- model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics_summary.csv    # Comparison table below
│   ├── best_params.json       # Tuned hyper-parameters
│   └── metadata.json          # Split, seed, versions
└── .streamlit/config.toml     # App theme
```

**How to reproduce**

```bash
pip install -r requirements.txt
python -m src.train_models      # retrains all 5 models (~1.5 min), rewrites model/ and test_data.csv
streamlit run app.py            # launches the app locally
```

---

## d. Models Used

All five models are scikit-learn `Pipeline`s (preprocessing + classifier). Hyper-parameters were tuned with
`GridSearchCV` (5-fold stratified CV, scoring = ROC-AUC) on the training set only. SVM was tuned on a stratified
8,000-row subsample (3-fold) because RBF-SVM training cost grows roughly quadratically with rows, then refit on
the full training set. Predictions use the default 0.5 decision threshold.

| Model | Key settings | Tuned hyper-parameters |
|---|---|---|
| Logistic Regression | L2, `max_iter=2000` | `C = 10` |
| Decision Tree | — | `criterion = entropy`, `max_depth = 6`, `min_samples_leaf = 100` |
| KNN | Euclidean distance | `n_neighbors = 75`, `weights = uniform` |
| Naive Bayes | Gaussian NB + Yeo-Johnson transform | `var_smoothing = 1e-3` |
| Support Vector Machine | RBF kernel | `C = 0.3`, `gamma = scale` |

### Comparison Table — evaluation metrics on the test set (5,993 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8046 | 0.7546 | 0.6285 | 0.2858 | 0.3929 | 0.3282 |
| Decision Tree | **0.8114** | **0.7640** | **0.6531** | 0.3152 | 0.4252 | 0.3597 |
| KNN | 0.8091 | 0.7625 | 0.6338 | 0.3250 | 0.4297 | 0.3556 |
| Naive Bayes | 0.7472 | 0.7457 | 0.4472 | **0.6033** | **0.5136** | 0.3550 |
| Support Vector Machine | 0.8113 | 0.7085 | 0.6462 | 0.3250 | 0.4325 | **0.3623** |

*Positive class = default (1). AUC for SVM is computed from its decision-function score (SVC trained without
Platt-scaling probabilities). Baseline: always predicting "no default" gives 0.779 accuracy and MCC = 0 — so
accuracy alone is misleading on this imbalanced dataset; MCC, F1 and AUC are the more informative metrics.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Solid ranking ability (AUC 0.755) but the **lowest Recall (0.286) and MCC (0.328)**: it catches fewer than 3 in 10 defaulters at the 0.5 threshold. The relationship between repayment status and default is threshold-like (risk jumps once a customer is ≥ 2 months late), which a single linear boundary approximates poorly. Its strength is transparency — coefficients are easy to explain to risk and audit teams. |
| **Decision Tree** | **Best AUC (0.764), Accuracy (0.811) and Precision (0.653)**, and MCC (0.360) within 0.003 of the top. Depth 6 with ≥ 100 samples per leaf keeps it from over-fitting. The most important splits are `PAY_0` (latest repayment status, 58 % importance) followed by the engineered `MAX_DELAY` (19 %) and `NUM_MONTHS_DELAYED` (4 %) — confirming that recent delinquency drives default risk and that the engineered features add value. |
| **KNN** | Very close to the Decision Tree (AUC 0.763, MCC 0.356). It needed a large neighbourhood (k = 75) because defaulters and non-defaulters overlap heavily in feature space, so small k mostly learns noise. Drawbacks: the model must store all 23,972 training rows (largest model file, ~2.7 MB), prediction is slow at scale, and it offers no explanation of *why* a customer is risky. |
| **Naive Bayes** | Operates at a very different trade-off: **highest Recall (0.603) and F1 (0.514)** but the lowest Accuracy (0.747) and Precision (0.447). The six `BILL_AMT` columns are strongly correlated (pairwise r ≥ 0.80), violating NB's independence assumption; the same evidence is counted several times, pushing more customers into the "default" class. Useful when missing a defaulter is far costlier than a false alarm, but it generates many false alarms. |
| **Support Vector Machine** | **Highest MCC (0.362)** and accuracy on par with the tree (0.811), but the **lowest AUC (0.709)** — its class predictions are strong, but ranking customers by its margin score is weaker than the other models. It is also the most expensive to train (~30 s per fit on 24k rows vs. < 1 s for the tree), had to be tuned on a subsample, and gives no native probabilities. |
| **Overall winner for this dataset** | **Decision Tree.** It has the best AUC, accuracy and precision, its MCC is statistically tied with SVM (0.360 vs 0.362), it trains and predicts fastest, and its rules are human-readable — important in a regulated credit-risk setting. If the business objective is to *catch as many defaulters as possible*, Naive Bayes (or the Decision Tree with a lower decision threshold) is preferable. Across all models, recall at the default 0.5 threshold is modest because only 22 % of customers default; threshold tuning or class weighting is the natural next step. |

---

## Streamlit App Features

| Requirement | Implementation |
|---|---|
| Dataset upload (CSV) | Sidebar file uploader (test data only) + option to use the bundled `test_data.csv` + download button for the sample file |
| Model selection dropdown | Sidebar `selectbox` with all 5 models and a one-line description of each |
| Display of evaluation metrics | 6 metric cards: Accuracy, AUC, Precision, Recall, F1, MCC |
| Confusion matrix / classification report | Annotated confusion-matrix heatmap (counts + %), full classification report, ROC curve |
| Extras | "Compare all models" tab (table, grouped bar chart, overlaid ROC curves), data preview with engineered features, column validation with clear error messages, predictions download |

The app loads the saved pipelines from `model/` with `joblib` and only calls `predict` /
`predict_proba` / `decision_function` — **no model is retrained on the uploaded data**.

**Uploaded CSV format:** must contain the 23 raw feature columns
(`LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2–PAY_6, BILL_AMT1–6, PAY_AMT1–6`) and the target
column `default.payment.next.month`. Column order does not matter; extra columns such as `ID` are ignored.

---

## Assumptions

1. Dataset obtained as `UCI_Credit_Card.csv` (Kaggle/UCI mirror, identical 30,000 × 25 content).
2. Undocumented category codes in `EDUCATION` and `MARRIAGE` are treated as "others".
3. Default decision threshold of 0.5 is used for all models so that the comparison is like-for-like.
4. Gaussian Naive Bayes was chosen over Multinomial NB because the features are continuous / ordinal, and
   some (`PAY_x`, `BILL_AMTx`) can be negative, which Multinomial NB does not accept.
5. Hyper-parameters were tuned on ROC-AUC (threshold-independent) rather than accuracy, because the classes
   are imbalanced.
