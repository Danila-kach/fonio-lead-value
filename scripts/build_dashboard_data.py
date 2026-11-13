"""Builds docs/data.js, the data behind the interactive dashboard.

Self-contained: trains the same two models as the notebook (Meta; Google Ads = Search + PMax)
on Oct 2025 - Jan 2026, scores the Feb - Mar 2026 test signups and writes them, together with
the model parameters the value calculator needs, into docs/data.js.
Needs only pandas, numpy and scikit-learn.   Run:  python scripts/build_dashboard_data.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = pd.Timestamp("2026-02-01")
CHANNEL_GROUPS = {"meta": ["meta"], "google": ["google_search", "google_pmax"]}
CATS, NUMS = ["country", "industry", "company_size", "device", "channel", "plan_by_h"], \
             ["business_email", "signup_hour", "signup_dow", "phone_connected_24h", "assistant_configured_24h", "test_calls_72h", "calendar_connected_72h", "paid_by_h"]

df = pd.read_csv(ROOT / "data" / "fonio_case_signups.csv")
df["signup_date"] = pd.to_datetime(df["signup_date"]); df["signup_dow"] = df["signup_date"].dt.day_of_week
for c in ["business_email", "phone_connected_24h", "assistant_configured_24h", "calendar_connected_72h", "converted_paid_30d"]:
    df[c] = df[c].astype(str).str.lower().eq("true").astype(int)
paid_by = df["days_to_first_payment"].le(3).fillna(False)          # what billing shows 72h after signup
df["paid_by_h"] = paid_by.astype(int); df["plan_by_h"] = df["plan_selected"].where(paid_by, "none")
X = pd.get_dummies(df[CATS + NUMS], columns=CATS, dtype=float)
y_paid, y_rev = df["converted_paid_30d"].values, df["revenue_180d_eur"].values
train = (df["signup_date"] < TRAIN_END).values


def fit(mask):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X[mask], y_paid[mask])
    payers = mask & (y_paid == 1)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(X[payers], y_rev[payers])
    return clf, reg, float(y_rev[payers].min()), float(y_rev[payers].mean())


def params(clf, reg, min_rev, mean_rev):
    sc1, lr, sc2, rg = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"], reg.named_steps["standardscaler"], reg.named_steps["ridge"]
    return {"columns": list(X.columns),
            "s1": {"mean": sc1.mean_.round(5).tolist(), "scale": sc1.scale_.round(5).tolist(), "coef": lr.coef_[0].round(5).tolist(), "intercept": float(lr.intercept_[0])},
            "s2": {"mean": sc2.mean_.round(5).tolist(), "scale": sc2.scale_.round(5).tolist(), "coef": rg.coef_.round(4).tolist(), "intercept": float(rg.intercept_)},
            "min_payer_revenue": min_rev, "mean_payer_revenue": round(mean_rev, 1)}


out = {"platforms": {}, "options": {k: sorted(df[k].unique().tolist()) for k in ["country", "industry", "company_size", "device"]}}
for g, label in [("meta", "Meta"), ("google", "Google Ads")]:
    in_group = df["channel"].isin(CHANNEL_GROUPS[g]).values
    trm, tem = train & in_group, ~train & in_group
    clf, reg, min_rev, mean_rev = fit(trm)
    p = clf.predict_proba(X[tem])[:, 1]; rip = np.clip(reg.predict(X[tem]), min_rev, None); value = p * rip
    rows = []
    for i, (_, r) in enumerate(df.loc[tem].iterrows()):
        rows.append({"id": r.signup_id, "date": r.signup_date.strftime("%Y-%m-%d"), "ch": r.channel, "co": r.country, "ind": r.industry, "cs": r.company_size,
                     "dev": r.device, "be": int(r.business_email), "ph": int(r.phone_connected_24h), "as": int(r.assistant_configured_24h),
                     "tc": int(r.test_calls_72h), "cal": int(r.calendar_connected_72h), "pb": int(r.paid_by_h), "pl": r.plan_by_h,
                     "v72": round(float(value[i]), 1), "p": round(float(p[i]), 3), "rip": round(float(rip[i]), 0),
                     "paid": int(r.converted_paid_30d), "rev": float(r.revenue_180d_eur)})
    out["platforms"][g] = {"label": label, "n_train": int(trm.sum()), "n_train_payers": int(y_paid[trm].sum()), "signups": rows, "model72": params(clf, reg, min_rev, mean_rev)}

(ROOT / "docs" / "data.js").write_text("window.FONIO = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("docs/data.js written,", round((ROOT / "docs" / "data.js").stat().st_size / 1024), "KB")
