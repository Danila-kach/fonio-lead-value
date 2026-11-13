"""Export everything the interactive dashboard needs into docs/data.js."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import load, feature_frame, split_by_time, channel_mask
from src.model import LeadValueModel

df = load(ROOT / "data" / "fonio_case_signups.csv")
tr, te = split_by_time(df)
y_paid, y_rev = df.converted_paid_30d.values, df.revenue_180d_eur.values
R = json.loads((ROOT / "outputs" / "results.json").read_text())
out = {"platforms": {}, "results": R["experiments"], "by_channel": R["tables"]["by_channel"], "behaviour_by_channel": R["tables"]["behaviour_by_channel"]}

def params(m):
    sc1 = m.clf.named_steps["standardscaler"]; lr = m.clf.named_steps["logisticregression"]
    sc2 = m.reg.named_steps["standardscaler"]; rg = m.reg.named_steps["ridge"]
    return {"columns": m.columns,
            "s1": {"mean": sc1.mean_.round(5).tolist(), "scale": sc1.scale_.round(5).tolist(), "coef": lr.coef_[0].round(5).tolist(), "intercept": float(lr.intercept_[0])},
            "s2": {"mean": sc2.mean_.round(5).tolist(), "scale": sc2.scale_.round(5).tolist(), "coef": rg.coef_.round(4).tolist(), "intercept": float(rg.intercept_)},
            "min_payer_revenue": m.min_payer_revenue, "mean_payer_revenue": round(m.mean_payer_revenue, 1)}

for g, label in [("meta", "Meta"), ("google", "Google Ads")]:
    trm, tem = tr & channel_mask(df, g), te & channel_mask(df, g)
    X72 = feature_frame(df, "72h")
    m72 = LeadValueModel().fit(X72[trm], y_paid[trm], y_rev[trm])
    ex = m72.explain(X72[tem])
    raw = df.loc[tem]
    rows = []
    for i, (idx, r) in enumerate(raw.iterrows()):
        rows.append({"id": r.signup_id, "date": r.signup_date.strftime("%Y-%m-%d"), "ch": r.channel, "co": r.country, "ind": r.industry, "cs": r.company_size,
                     "dev": r.device, "be": int(r.business_email), "ph": int(r.phone_connected_24h), "as": int(r.assistant_configured_24h),
                     "tc": int(r.test_calls_72h), "cal": int(r.calendar_connected_72h), "pb": int(pd.notna(r.days_to_first_payment) and r.days_to_first_payment <= 3),
                     "pl": r.plan_selected if (pd.notna(r.days_to_first_payment) and r.days_to_first_payment <= 3) else "none",
                     "v72": round(float(ex.loc[idx, "value"]), 1), "p": round(float(ex.loc[idx, "p_pay"]), 3), "rip": round(float(ex.loc[idx, "revenue_if_pays"]), 0),
                     "paid": int(r.converted_paid_30d), "rev": float(r.revenue_180d_eur)})
    out["platforms"][g] = {"label": label, "n_train": int(trm.sum()), "n_train_payers": int(y_paid[trm].sum()), "signups": rows,
                           "model72": params(m72)}
# option lists for the calculator
out["options"] = {k: sorted(df[k].unique().tolist()) for k in ["country", "industry", "company_size", "device"]}
(ROOT / "docs" / "data.js").write_text("window.FONIO = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("data.js", round((ROOT / "docs" / "data.js").stat().st_size / 1024), "KB")
