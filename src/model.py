"""Lead-value model: expected 180-day revenue of a signup, scored at 72h.

Two-stage ("hurdle") model:
  stage 1  P(pays within 30 days)         -> logistic regression (or GBM)
  stage 2  E[revenue_180d | pays]          -> ridge regression on payers
  value    = P(pays) * E[revenue | pays]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LeadValueModel:
    def __init__(self, stage1: str = "logreg", stage2: str = "ridge"):
        self.stage1_name, self.stage2_name = stage1, stage2
        if stage1 == "logreg":
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        else:
            self.clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                      max_iter=300, l2_regularization=1.0)
        if stage2 == "ridge":
            self.reg = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        elif stage2 == "gbm":
            self.reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200)
        else:
            self.reg = None  # variant A: constant mean revenue of payers
        self.columns: list[str] | None = None
        self.mean_payer_revenue = np.nan
        self.min_payer_revenue = 0.0  # floor for stage 2: a payer never brings less than one month of the cheapest plan

    def fit(self, X: pd.DataFrame, y_paid: np.ndarray, y_rev: np.ndarray):
        self.columns = list(X.columns)
        self.clf.fit(X, y_paid)
        payers = y_paid == 1
        self.mean_payer_revenue = float(y_rev[payers].mean())
        self.min_payer_revenue = float(y_rev[payers].min())
        if self.reg is not None:
            self.reg.fit(X[payers], y_rev[payers])
        return self

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.columns, fill_value=0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._align(X))[:, 1]

    def predict_value(self, X: pd.DataFrame) -> np.ndarray:
        X = self._align(X)
        p = self.clf.predict_proba(X)[:, 1]
        if self.reg is None:
            rev = np.full(len(X), self.mean_payer_revenue)
        else:
            rev = np.clip(self.reg.predict(X), self.min_payer_revenue, None)
        return p * rev

    def coefficients(self) -> pd.Series | None:
        """Stage 1 weights ("will they pay?"). Positive = more likely to pay."""
        if self.stage1_name != "logreg":
            return None
        lr = self.clf.named_steps["logisticregression"]
        return pd.Series(lr.coef_[0], index=self.columns).sort_values()

    def coefficients_stage2(self) -> pd.Series | None:
        """Stage 2 weights ("if they pay, how much?"), in euros per standardised unit."""
        if self.stage2_name != "ridge":
            return None
        rg = self.reg.named_steps["ridge"]
        return pd.Series(rg.coef_, index=self.columns).sort_values()

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-lead breakdown: probability, revenue-if-pays, value."""
        X = self._align(X)
        p = self.clf.predict_proba(X)[:, 1]
        rev = np.full(len(X), self.mean_payer_revenue) if self.reg is None else np.clip(self.reg.predict(X), self.min_payer_revenue, None)
        return pd.DataFrame({"p_pay": p, "revenue_if_pays": rev, "value": p * rev}, index=X.index)


class RuleBaseline:
    """What a marketer could do by hand: a lookup table of mean revenue by
    (phone connected 24h, calendar connected 72h)."""

    def __init__(self, keys=("phone_connected_24h", "calendar_connected_72h")):
        self.keys = list(keys)
        self.table = None
        self.global_mean = np.nan

    def fit(self, df: pd.DataFrame, y_rev: np.ndarray):
        d = df[self.keys].copy()
        d["rev"] = y_rev
        self.table = d.groupby(self.keys)["rev"].mean()
        self.global_mean = float(np.mean(y_rev))
        return self

    def predict_value(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[self.keys])
        return self.table.reindex(idx).fillna(self.global_mean).values
