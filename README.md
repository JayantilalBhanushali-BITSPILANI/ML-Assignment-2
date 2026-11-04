# Credit Card Default Prediction using 5 Classification Models

ML Assignment 2, M.Tech (AIML), BITS Pilani WILP

- Live Streamlit App: https://jayantilal-ml-assignment-2.streamlit.app/
- GitHub Repository: https://github.com/JayantilalBhanushali-BITSPILANI/ML-Assignment-2

---

## a. Problem Statement

The aim is to predict whether a credit card customer will default on the next month's payment. This is a binary classification problem (1 = default, 0 = no default). The inputs are the customer's credit limit, demographic details, and the repayment status, bill amount and payment amount for the last 6 months.

I trained five models on the same dataset and the same train/test split: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM. Each model is evaluated using Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are saved and used in a Streamlit app, which calculates the metrics on an uploaded test CSV without retraining.

---

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Default of Credit Card Clients |
| Source | UCI Machine Learning Repository (ID 350): https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. Also available on Kaggle (uciml/default-of-credit-card-clients-dataset) |
| Reference | Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. |
| Instances | 30,000 customers from Taiwan (April to September 2005). 29,965 after removing 35 duplicate rows |
| Input features | 23 (minimum required is 12) |
| Features used in models | 28 (23 original + 5 created by me) |
| Target column | default.payment.next.month (1 = default, 0 = no default) |
| Class balance | 77.9% no default, 22.1% default (imbalanced) |
| Missing values | None |

Feature groups:

| Group | Columns | Meaning |
|---|---|---|
| Credit limit | LIMIT_BAL | Credit given in NT dollars, including supplementary cards |
| Demographics | SEX, EDUCATION, MARRIAGE, AGE | SEX: 1 = male, 2 = female. EDUCATION: 1 = graduate school, 2 = university, 3 = high school, 4 = others. MARRIAGE: 1 = married, 2 = single, 3 = others |
| Repayment status | PAY_0, PAY_2 to PAY_6 | September back to April 2005. -2, -1, 0 = paid on time or no use; 1 to 9 = months of delay |
| Bill amounts | BILL_AMT1 to BILL_AMT6 | Monthly bill amount (NT dollars), September back to April |
| Payment amounts | PAY_AMT1 to PAY_AMT6 | Amount paid (NT dollars), September back to April |

Preprocessing and feature engineering (in src/features.py, used by both the training script and the app):

1. Removed the ID column and 35 duplicate rows, so the same record cannot appear in both train and test.
2. EDUCATION has some codes (0, 5, 6) that are not explained in the dataset documentation, so I grouped them into 4 (others). Similarly, MARRIAGE = 0 was grouped into 3 (others).
3. Created 5 new features:
    - AVG_UTILIZATION: average bill amount divided by credit limit
    - LATEST_UTILIZATION: latest bill amount divided by credit limit
    - PAYMENT_RATIO: total amount paid divided by total amount billed (limited to the range 0 to 2)
    - NUM_MONTHS_DELAYED: number of months in which the payment was delayed
    - MAX_DELAY: the longest delay in the last 6 months
4. Each model pipeline has its own preprocessing: one-hot encoding for SEX, EDUCATION and MARRIAGE; StandardScaler for Logistic Regression, KNN and SVM; PowerTransformer (Yeo-Johnson) for Naive Bayes; and no scaling for the Decision Tree.
5. Stratified 80/20 train/test split with random_state = 2026, giving 23,972 training rows and 5,993 test rows. The test rows are saved as test_data.csv in the original column format.

---

## c. GitHub Repository Link

https://github.com/JayantilalBhanushali-BITSPILANI/ML-Assignment-2

Repository structure:

```
ML-Assignment-2/
|-- app.py                     Streamlit app (loads saved models, no retraining)
|-- requirements.txt           Python libraries with versions
|-- README.md                  This file
|-- test_data.csv              Test set (5,993 rows, 23 features + target)
|-- data/
|   |-- UCI_Credit_Card.csv    Full dataset (30,000 rows)
|-- src/
|   |-- features.py            Data cleaning and feature engineering
|   |-- train_models.py        Training, tuning, evaluation and saving models
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- metrics_summary.csv    Metrics table shown below
|   |-- best_params.json       Selected hyperparameters
|   |-- metadata.json          Split details, random seed, library version
|-- .streamlit/config.toml     App theme settings
```

How to run:

```
pip install -r requirements.txt
python -m src.train_models     (retrains all 5 models, takes about 2 minutes)
streamlit run app.py           (starts the app locally)
```

---

## d. Models Used

All five models are built as scikit-learn Pipelines (preprocessing + classifier). Hyperparameters were selected with GridSearchCV using 5-fold stratified cross validation on the training data, with ROC-AUC as the scoring metric. For SVM, the grid search was run on a stratified sample of 8,000 training rows with 3 folds, because SVM training becomes very slow on large data. The best setting was then trained on the full training set. All predictions use the default threshold of 0.5.

| Model | Settings | Selected hyperparameters |
|---|---|---|
| Logistic Regression | L2 penalty, max_iter = 2000 | C = 10 |
| Decision Tree | - | criterion = entropy, max_depth = 6, min_samples_leaf = 100 |
| KNN | Euclidean distance | n_neighbors = 75, weights = uniform |
| Naive Bayes | GaussianNB with Yeo-Johnson transform | var_smoothing = 0.001 |
| Support Vector Machine | RBF kernel | C = 0.3, gamma = scale |

### Comparison table (test set, 5,993 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8046 | 0.7546 | 0.6285 | 0.2858 | 0.3929 | 0.3282 |
| Decision Tree | 0.8114 | 0.7640 | 0.6531 | 0.3152 | 0.4252 | 0.3597 |
| KNN | 0.8091 | 0.7625 | 0.6338 | 0.3250 | 0.4297 | 0.3556 |
| Naive Bayes | 0.7472 | 0.7457 | 0.4472 | 0.6033 | 0.5136 | 0.3550 |
| Support Vector Machine | 0.8113 | 0.7085 | 0.6462 | 0.3250 | 0.4325 | 0.3623 |

Notes:

- The positive class is default (1).
- The SVM was trained without probability output, so its AUC is calculated from the decision function score.
- If a model simply predicted "no default" for everyone, it would still get 0.779 accuracy but MCC = 0. So on this dataset, accuracy alone is not a good measure. MCC, F1 and AUC give a better picture.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | AUC is reasonable (0.755), but it has the lowest recall (0.286) and the lowest MCC (0.328), which means it identifies less than 3 out of 10 defaulters. Default risk increases sharply once a customer is 2 or more months late, and a linear model cannot capture this kind of jump well. Its advantage is that the coefficients are easy to explain. |
| Decision Tree | Has the highest AUC (0.764), accuracy (0.811) and precision (0.653). Its MCC (0.360) is almost the same as the best (0.362). Limiting the depth to 6 and keeping at least 100 samples per leaf prevented overfitting. The most important feature is PAY_0, the latest repayment status (58% importance), followed by two of my created features, MAX_DELAY (19%) and NUM_MONTHS_DELAYED (4%). So recent payment delays are the main sign of default, and the new features were useful. |
| KNN | Results are very close to the Decision Tree (AUC 0.763, MCC 0.356). The best k was large (75), because defaulters and non-defaulters overlap a lot, and with a small k the model was fitting noise. The disadvantages are that it has to store the full training data (the largest model file, about 2.7 MB), prediction is slow on large data, and it does not explain its predictions. |
| Naive Bayes | Behaves differently from the other models. It has the highest recall (0.603) and F1 (0.514), but the lowest accuracy (0.747) and precision (0.447). The six BILL_AMT columns are highly correlated with each other (correlation of 0.80 or more), but Naive Bayes assumes the features are independent, so the same information is counted multiple times. As a result it predicts "default" much more often. This is useful if missing a defaulter is very costly, but it also gives many false alarms. |
| Support Vector Machine | Has the highest MCC (0.362) and accuracy similar to the Decision Tree (0.811), but the lowest AUC (0.709). So its final class predictions are good, but the ranking of customers by its score is weaker than the other models. It was also the slowest to train (about 30 seconds per fit, compared to less than 1 second for the Decision Tree), so the tuning had to be done on a sample. |
| Overall winner | Decision Tree. It has the best AUC, accuracy and precision, its MCC is practically the same as SVM (0.360 vs 0.362), it is fast, and its rules can be easily explained, which matters for credit decisions in banking. If the main goal is to catch as many defaulters as possible, Naive Bayes, or the Decision Tree with a lower threshold, would be a better choice. For all models, recall at the 0.5 threshold is low because only 22% of customers default. Adjusting the threshold or using class weights could improve this. |

---

## Streamlit App Features

| Requirement | What the app does |
|---|---|
| Dataset upload (CSV) | File upload option in the sidebar for test data. There is also an option to use the included test_data.csv and a button to download it |
| Model selection dropdown | Dropdown in the sidebar with all 5 models and a short description of each |
| Evaluation metrics | Shows Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model |
| Confusion matrix and classification report | Confusion matrix heatmap (counts and percentages), classification report and ROC curve |
| Additional | A tab comparing all 5 models (table, bar chart and ROC curves), data preview, a check for missing columns with an error message, and download of predictions |

The app loads the saved models from the model folder using joblib and only uses them for prediction. No model is retrained on the uploaded data.

Uploaded CSV format: the file must contain the 23 input columns (LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2 to PAY_6, BILL_AMT1 to BILL_AMT6, PAY_AMT1 to PAY_AMT6) and the target column default.payment.next.month. The column order does not matter, and extra columns such as ID are ignored.

---

## Assumptions

1. The dataset was downloaded as UCI_Credit_Card.csv (Kaggle copy of the UCI dataset, 30,000 rows x 25 columns).
2. The undocumented codes in EDUCATION and MARRIAGE are treated as "others".
3. A threshold of 0.5 is used for all models so that the comparison is fair.
4. Gaussian Naive Bayes was used instead of Multinomial Naive Bayes because the features are continuous and some of them (PAY_x, BILL_AMTx) have negative values, which Multinomial NB does not allow.
5. ROC-AUC was used for hyperparameter tuning instead of accuracy, because the classes are imbalanced.
