"""Marketer-facing evaluation: gain curve, tertiles, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gain_curve(score: np.ndarray, revenue: np.ndarray, points: int = 100):
    order = np.argsort(-score, kind="stable")
    rev = revenue[order]
    cum = np.cumsum(rev) / rev.sum()
    share = np.arange(1, len(rev) + 1) / len(rev)
    grid = np.linspace(0, 1, points + 1)[1:]
    return grid, np.interp(grid, share, cum)


def revenue_captured(score, revenue, top_share=0.3) -> float:
    grid, cum = gain_curve(score, revenue, points=1000)
    return float(np.interp(top_share, grid, cum))


def tertile_table(score, paid, revenue) -> pd.DataFrame:
    q = pd.qcut(pd.Series(score).rank(method="first"), 3, labels=["bottom", "middle", "top"])
    d = pd.DataFrame({"tertile": q, "paid": paid, "revenue": revenue, "score": score})
    t = d.groupby("tertile", observed=True).agg(
        leads=("paid", "size"), paid_rate=("paid", "mean"),
        revenue_per_lead=("revenue", "mean"), predicted_per_lead=("score", "mean"))
    return t.loc[["top", "middle", "bottom"]]


def calibration_table(score, revenue, bins: int = 5) -> pd.DataFrame:
    q = pd.qcut(pd.Series(score).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"bin": q, "pred": score, "actual": revenue})
    return d.groupby("bin").agg(leads=("pred", "size"), predicted=("pred", "mean"),
                                actual=("actual", "mean"))


def summary(score, paid, revenue) -> dict:
    return {
        "n": int(len(score)),
        "auc_paid": float(roc_auc_score(paid, score)) if len(set(paid)) > 1 else float("nan"),
        "revenue_top30_share": revenue_captured(score, revenue, 0.3),
        "revenue_top50_share": revenue_captured(score, revenue, 0.5),
        "predicted_mean": float(np.mean(score)),
        "actual_mean": float(np.mean(revenue)),
    }
