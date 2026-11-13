"""Feature construction for the lead-value model.

Every feature is built from what the product database would show at the
scoring moment (24h or 72h after signup). Columns that are only known
after the outcome (plan_selected, days_to_first_payment, converted_paid_30d,
revenue_180d_eur) are never used directly; the only derived signal is
"already paid by the scoring moment", which is genuinely visible at that time.
"""
from __future__ import annotations

import pandas as pd

HORIZON_DAYS = {"24h": 1, "72h": 3}

BASE_CATEGORICAL = ["country", "industry", "company_size", "device", "channel"]
BASE_NUMERIC = ["business_email", "signup_hour", "signup_dow"]

FEATURES_BY_HORIZON = {
    "24h": ["phone_connected_24h", "assistant_configured_24h", "paid_by_h", "plan_by_h"],
    "72h": [
        "phone_connected_24h", "assistant_configured_24h",
        "test_calls_72h", "calendar_connected_72h",
        "paid_by_h", "plan_by_h",
    ],
}

# Known only after the outcome, or without a timestamp in the dataset.
EXCLUDED = {
    "plan_selected": "outcome of the payment; at 72h it is 'none' for 90% of future payers",
    "days_to_first_payment": "outcome; only used to derive 'paid_by_h'",
    "converted_paid_30d": "target",
    "revenue_180d_eur": "target",
    "minutes_used_7d": "needs 7 days; not available at 72h",
    "onboarding_call_booked": "no timestamp in the dataset; cannot be placed before/after 72h",
    "signup_id": "identifier",
}

TRAIN_END = pd.Timestamp("2026-02-01")  # train: Oct-Jan, test: Feb-Mar

CHANNEL_GROUPS = {
    "meta": ["meta"],
    "google": ["google_search", "google_pmax"],
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_dow"] = df["signup_date"].dt.dayofweek
    for c in ["business_email", "phone_connected_24h", "assistant_configured_24h",
              "calendar_connected_72h", "onboarding_call_booked", "converted_paid_30d"]:
        df[c] = df[c].astype(str).str.lower().eq("true").astype(int)
    return df


def add_horizon_features(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Derive what is visible at the scoring moment."""
    days = HORIZON_DAYS[horizon]
    out = df.copy()
    paid_by = out["days_to_first_payment"].le(days).fillna(False)
    out["paid_by_h"] = paid_by.astype(int)
    out["plan_by_h"] = out["plan_selected"].where(paid_by, "none")
    return out


def feature_frame(df: pd.DataFrame, horizon: str, include_channel: bool = True) -> pd.DataFrame:
    df = add_horizon_features(df, horizon)
    cats = [c for c in BASE_CATEGORICAL if include_channel or c != "channel"] + ["plan_by_h"]
    nums = BASE_NUMERIC + [c for c in FEATURES_BY_HORIZON[horizon] if c != "plan_by_h"]
    X = pd.get_dummies(df[cats + nums], columns=cats, dtype=float)
    return X


def split_by_time(df: pd.DataFrame):
    train = df["signup_date"] < TRAIN_END
    return train.values, (~train).values


def channel_mask(df: pd.DataFrame, group: str):
    return df["channel"].isin(CHANNEL_GROUPS[group]).values
