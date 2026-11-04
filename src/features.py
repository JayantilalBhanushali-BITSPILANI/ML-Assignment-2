"""
features.py
-----------
Single source of truth for data cleaning and feature engineering.

The SAME function (`prepare_features`) is used by:
  * src/train_models.py  -> while training the models
  * app.py               -> while scoring the uploaded test CSV in Streamlit

Keeping it in one place guarantees that training-time and inference-time
transformations can never drift apart (a common cause of silent bugs in
deployed ML apps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
TARGET_COL = "default.payment.next.month"
ID_COL = "ID"

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# The 23 raw input features expected in any uploaded CSV
RAW_FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_STATUS_COLS
    + BILL_COLS
    + PAY_AMT_COLS
)

# Engineered features (created by `prepare_features`)
ENGINEERED_FEATURES = [
    "AVG_UTILIZATION",     # average bill / credit limit over 6 months
    "LATEST_UTILIZATION",  # most recent bill / credit limit
    "PAYMENT_RATIO",       # total paid / total billed over 6 months
    "NUM_MONTHS_DELAYED",  # count of months with a payment delay (PAY_x > 0)
    "MAX_DELAY",           # worst delay seen in the last 6 months
]

# Nominal categoricals -> one-hot encoded inside the model pipelines
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

# Everything else is treated as numeric
NUMERIC_FEATURES = [c for c in RAW_FEATURES if c not in CATEGORICAL_FEATURES] + ENGINEERED_FEATURES

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # 28 columns in total


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def validate_columns(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Return the list of required columns that are missing from `df`."""
    required = RAW_FEATURES + ([TARGET_COL] if require_target else [])
    return [c for c in required if c not in df.columns]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented category codes present in the original UCI data.

    * EDUCATION: documented codes are 1-4; values 0, 5, 6 are undocumented
      -> merged into 4 ("others").
    * MARRIAGE: documented codes are 1-3; value 0 is undocumented
      -> merged into 3 ("others").
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 3)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 domain-driven credit-risk features (all row-wise, no leakage)."""
    df = df.copy()
    limit = df["LIMIT_BAL"].replace(0, np.nan)

    total_bill = df[BILL_COLS].sum(axis=1)
    total_paid = df[PAY_AMT_COLS].sum(axis=1)

    df["AVG_UTILIZATION"] = (df[BILL_COLS].mean(axis=1) / limit).fillna(0)
    df["LATEST_UTILIZATION"] = (df["BILL_AMT1"] / limit).fillna(0)
    # Ratio is clipped to [0, 2] so customers with ~zero bills don't create extreme outliers
    df["PAYMENT_RATIO"] = (total_paid / total_bill.where(total_bill > 0, np.nan)).fillna(1.0).clip(0, 2)
    df["NUM_MONTHS_DELAYED"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)
    df["MAX_DELAY"] = df[PAY_STATUS_COLS].max(axis=1).clip(lower=0)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> model-ready feature matrix (28 columns, fixed order)."""
    df = clean_raw(df)
    df = add_engineered_features(df)
    return df[MODEL_FEATURES]
