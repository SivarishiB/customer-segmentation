"""Shared data cleaning and feature engineering.

Used by the notebooks AND the Streamlit app, so training and prediction
always transform customers in exactly the same way. Change features here only
after the team agrees (feature list is frozen at the end of Week 1).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "marketing_campaign.xlsx"
MODEL_PATH = ROOT / "app" / "model.pkl"

# The data covers customers enrolled Jul 2012 - Jun 2014, so ages are measured in 2014.
REFERENCE_YEAR = 2014
RANDOM_STATE = 42

MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                 "AcceptedCmp4", "AcceptedCmp5", "Response"]

# Raw columns a new customer needs (app input / CSV upload).
RAW_INPUT_COLS = (["Year_Birth", "Education", "Marital_Status", "Income", "Kidhome",
                   "Teenhome", "Dt_Customer", "Recency"] + MNT_COLS +
                  ["NumDealsPurchases"] + PURCHASE_COLS + ["NumWebVisitsMonth"] +
                  CAMPAIGN_COLS)

EDUCATION_LEVEL = {"Basic": 0, "2n Cycle": 0, "Graduation": 1, "Master": 2, "PhD": 2}
PARTNER_STATUSES = ["Married", "Together"]   # everything else (incl. YOLO/Absurd/Alone) = living alone

# Features used for clustering.
FEATURES = ["Income", "Age", "Children", "Family_Size", "Education_Level", "Recency",
            "Total_Spent", "Total_Purchases", "NumDealsPurchases", "NumWebVisitsMonth",
            "Total_Accepted", "Customer_Days", "MntWines", "MntMeatProducts",
            "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the original Excel file."""
    return pd.read_excel(path)


def clean(df: pd.DataFrame, income_median: float | None = None) -> pd.DataFrame:
    """Fix missing income, parse dates, drop constant columns."""
    df = df.copy()
    if income_median is None:
        income_median = df["Income"].median()
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce").fillna(income_median)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    return df.drop(columns=["Z_CostContact", "Z_Revenue"], errors="ignore")


def add_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Create the engineered features. `reference_date` = latest enrolment date in training data."""
    df = df.copy()
    if reference_date is None:
        reference_date = df["Dt_Customer"].max()
    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["Total_Spent"] = df[MNT_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["Total_Accepted"] = df[CAMPAIGN_COLS].sum(axis=1)
    df["Customer_Days"] = (pd.Timestamp(reference_date) - df["Dt_Customer"]).dt.days.clip(lower=0)
    df["Partner"] = df["Marital_Status"].isin(PARTNER_STATUSES).astype(int)
    df["Family_Size"] = 1 + df["Partner"] + df["Children"]
    df["Education_Level"] = df["Education"].map(EDUCATION_LEVEL).fillna(1).astype(int)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible ages (birth year 1893-1900) and the 666,666 income."""
    return df[(df["Age"] < 90) & (df["Income"] < 200_000)].reset_index(drop=True)


def load_clean_data(path=DATA_PATH) -> pd.DataFrame:
    """Full pipeline: raw Excel -> cleaned, feature-engineered, outlier-free data (2,236 rows)."""
    return remove_outliers(add_features(clean(load_raw(path))))
