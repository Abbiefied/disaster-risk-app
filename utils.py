"""
utils.py - Shared helper functions for the Disaster Risk Intelligence System.
All model artifacts are loaded once here and imported by app.py.
"""

import numpy as np
import pandas as pd
import joblib
from config import *

#Load artifacts once at import time
model           = joblib.load(MODEL_PATH)
scaler          = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a raw country-year row (or constructed next-year row) into a
    model-ready DataFrame:
      1. Add missing feature columns (fill with 0)
      2. Reorder columns to match training order
      3. Scale the numeric columns using the fitted scaler
    """
    df = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    scale_cols_present = [c for c in SCALE_COLS if c in df.columns]
    df[scale_cols_present] = scaler.transform(df[scale_cols_present])

    return df


def predict_risk(input_df: pd.DataFrame):
    """
    Returns (prediction: int, probability: float) for a prepared input row.
    prediction = 1 → High Occurrence, 0 → Low Occurrence
    probability = P(High Occurrence)
    """
    X    = prepare_input(input_df)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return pred, prob


def build_next_year_input(current_row: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a synthetic next-year feature row from the current year's data.

    Roll the current year's actuals forward:
      • prev_year_count         ← current year's event_count
      • log_prev_total_deaths   ← log1p(current year's total_deaths)
      • log_prev_total_affected ← log1p(current year's total_affected)
      • log_prev_total_damage   ← log1p(current year's total_damage)

    Type-count columns (flood_count, storm_count, etc.) are kept as-is;
    they represent the historical type composition, which is a reasonable
    proxy for the expected composition in t+1 when no forecast data exists.
    """
    nxt = current_row.copy()

    #Advance the year
    nxt["Start_Year"] = current_row["Start_Year"].values[0] + 1

    #Roll event_count into prev_year_count
    if "event_count" in current_row.columns:
        nxt["prev_year_count"] = current_row["event_count"].values[0]

    #Roll raw impact columns → log-transformed lag columns
    for lag_col, raw_col in RAW_IMPACT_COLS.items():
        if raw_col in current_row.columns and lag_col in feature_columns:
            nxt[lag_col] = np.log1p(current_row[raw_col].values[0])

    return nxt


def get_feature_importance() -> pd.DataFrame:
    """
    Return a DataFrame of logistic regression coefficients sorted by absolute
    magnitude.  Used in the app's 'What drives risk?' panel.
    """
    coefs = model.coef_[0]
    df = pd.DataFrame({
        "Feature"    : feature_columns,
        "Coefficient": coefs,
    })
    df["abs_coef"] = df["Coefficient"].abs()
    df["Direction"] = df["Coefficient"].apply(
        lambda c: "Increases Risk" if c > 0 else "Decreases Risk"
    )
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def get_risk_drivers(input_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    For a single prediction row, compute each feature's contribution to the
    log-odds score (coefficient × scaled feature value).  Positive contributions
    push toward High Risk; negative push toward Low Risk.

    Returns the top_n drivers sorted by absolute contribution.
    """
    X_scaled   = prepare_input(input_df)
    coefs      = model.coef_[0]
    values     = X_scaled.values[0]
    contrib    = coefs * values

    df = pd.DataFrame({
        "Feature"     : feature_columns,
        "Contribution": contrib,
    })
    df["Direction"] = df["Contribution"].apply(
        lambda c: "↑ Raises Risk" if c > 0 else "↓ Lowers Risk"
    )
    df["abs"] = df["Contribution"].abs()
    return (
        df.sort_values("abs", ascending=False)
          .head(top_n)
          .drop(columns="abs")
          .reset_index(drop=True)
    )


def risk_label(prob: float) -> tuple[str, str]:
    """Return (label, hex colour) for a given probability."""
    if prob >= 0.75:
        return "Very High", "#e74c3c"
    if prob >= 0.55:
        return "High",      "#e67e22"
    if prob >= 0.40:
        return "Moderate",  "#f1c40f"
    return "Low", "#2ecc71"