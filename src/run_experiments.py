"""Run all experiments and dump numbers to outputs/results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev
from src.features import channel_mask, feature_frame, load, split_by_time
from src.model import LeadValueModel, RuleBaseline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def run(df, horizon, train_group, test_group, stage1="logreg", stage2="ridge", include_channel=True):
    X = feature_frame(df, horizon, include_channel=include_channel)
    tr, te = split_by_time(df)
    trm = tr & channel_mask(df, train_group) if train_group != "all" else tr
    tem = te & channel_mask(df, test_group)
    y_paid = df["converted_paid_30d"].values
    y_rev = df["revenue_180d_eur"].values
    m = LeadValueModel(stage1, stage2).fit(X[trm], y_paid[trm], y_rev[trm])
    score = m.predict_value(X[tem])
    res = ev.summary(score, y_paid[tem], y_rev[tem])
    res.update(dict(horizon=horizon, train=train_group, test=test_group,
                    stage1=stage1, stage2=stage2, include_channel=include_channel,
                    n_train=int(trm.sum()), n_train_payers=int(y_paid[trm].sum())))
    return m, score, tem, res


def main():
    df = load(ROOT / "data" / "fonio_case_signups.csv")
    y_paid = df["converted_paid_30d"].values
    y_rev = df["revenue_180d_eur"].values
    tr, te = split_by_time(df)
    results = {"experiments": [], "tables": {}}

    # --- main models: per platform, 72h, logreg + ridge ---
    main_scores = {}
    for g in ["meta", "google"]:
        m, score, tem, res = run(df, "72h", g, g)
        res["name"] = f"{g}_72h_main"
        results["experiments"].append(res)
        main_scores[g] = (score, tem)
        results["tables"][f"tertiles_{g}"] = ev.tertile_table(score, y_paid[tem], y_rev[tem]).reset_index().to_dict("records")
        results["tables"][f"calibration_{g}"] = ev.calibration_table(score, y_rev[tem]).reset_index().to_dict("records")
        coef = m.coefficients()
        results["tables"][f"coefficients_{g}"] = coef.round(3).to_dict()
        results["tables"][f"coefficients_stage2_{g}"] = m.coefficients_stage2().round(1).to_dict()
        results["tables"][f"mean_payer_revenue_{g}"] = round(m.mean_payer_revenue, 1)
        # two example leads from the hold-out: highest and lowest predicted value
        X = feature_frame(df, "72h"); ex = m.explain(X[tem]); raw = df.loc[tem]
        show = ["channel", "country", "industry", "company_size", "device", "business_email", "phone_connected_24h",
                "assistant_configured_24h", "test_calls_72h", "calendar_connected_72h", "converted_paid_30d", "revenue_180d_eur"]
        examples = []
        not_paid_yet = raw["days_to_first_payment"].isna() | raw["days_to_first_payment"].gt(3)
        exv = ex.loc[not_paid_yet.values, "value"]  # examples the model judged on behaviour alone, not on an early payment
        for label, idx in [("high", exv.idxmax()), ("median", exv.sort_values().index[len(exv) // 2]), ("low", exv.idxmin())]:
            rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in raw.loc[idx, show].items()}
            rec.update(label=label, p_pay=round(float(ex.loc[idx, "p_pay"]), 3), revenue_if_pays=round(float(ex.loc[idx, "revenue_if_pays"]), 0),
                       value=round(float(ex.loc[idx, "value"]), 0))
            examples.append(rec)
        results["tables"][f"examples_{g}"] = examples
        # what the signups in each third look like (shares of the third)
        t = pd.qcut(pd.Series(score).rank(method="first"), 3, labels=["bottom", "middle", "top"]).values
        prof = raw.assign(third=t, paid_by_72h=raw["days_to_first_payment"].le(3), solo=raw["company_size"].eq("1")).groupby("third", observed=True).agg(
            signups=("signup_id", "size"), value_sent=("signup_id", lambda i: float(np.mean(score[np.isin(raw["signup_id"], i)]))),
            phone_connected_24h=("phone_connected_24h", "mean"), assistant_configured_24h=("assistant_configured_24h", "mean"),
            calendar_connected_72h=("calendar_connected_72h", "mean"), test_calls_72h=("test_calls_72h", "mean"),
            business_email=("business_email", "mean"), paid_by_72h=("paid_by_72h", "mean"), solo_company=("solo", "mean"),
            paid_30d=("converted_paid_30d", "mean"), revenue_180d=("revenue_180d_eur", "mean"))
        results["tables"][f"third_profile_{g}"] = prof.round(3).reset_index().to_dict("records")
        # gain curve points
        grid, cum = ev.gain_curve(score, y_rev[tem], points=20)
        results["tables"][f"gain_{g}"] = {"share": grid.round(3).tolist(), "captured": cum.round(4).tolist()}

        # baselines on the same test set
        base = RuleBaseline().fit(df[tr & channel_mask(df, g)], y_rev[tr & channel_mask(df, g)])
        bscore = base.predict_value(df[tem])
        bres = ev.summary(bscore, y_paid[tem], y_rev[tem]); bres.update(name=f"{g}_rule_baseline", horizon="72h", train=g, test=g)
        results["experiments"].append(bres)
        rng = np.random.default_rng(0)
        rscore = rng.random(tem.sum())
        rres = ev.summary(rscore, y_paid[tem], y_rev[tem]); rres.update(name=f"{g}_all_equal", horizon="-", train=g, test=g)
        results["experiments"].append(rres)

        # 24h vs 72h
        for h in ["24h"]:
            _, s, _, r = run(df, h, g, g); r["name"] = f"{g}_{h}"; results["experiments"].append(r)
        # pooled model (all channels, channel as feature) scored on this platform's leads
        _, s, _, r = run(df, "72h", "all", g, include_channel=True); r["name"] = f"{g}_pooled_with_channel"; results["experiments"].append(r)
        _, s, _, r = run(df, "72h", "all", g, include_channel=False); r["name"] = f"{g}_pooled_no_channel"; results["experiments"].append(r)
        # model variants
        _, s, _, r = run(df, "72h", g, g, stage2="const"); r["name"] = f"{g}_variantA_const_revenue"; results["experiments"].append(r)
        _, s, _, r = run(df, "72h", g, g, stage1="gbm", stage2="gbm"); r["name"] = f"{g}_gbm"; results["experiments"].append(r)

    # Google: one model for Search + PMax vs one model each
    rows = []
    gm_all = run(df, "72h", "google", "google")[0]
    for ch in ["google_search", "google_pmax"]:
        cm = df["channel"].eq(ch).values
        trm, tem = tr & cm, te & cm
        X = feature_frame(df, "72h")
        own = LeadValueModel().fit(X[trm], y_paid[trm], y_rev[trm]).predict_value(X[tem])
        joint = gm_all.predict_value(X[tem])
        for name, s_ in [("own model", own), ("one Google Ads model", joint)]:
            r = ev.summary(s_, y_paid[tem], y_rev[tem]); r.update(channel=ch, model=name, n_train=int(trm.sum()), n_train_payers=int(y_paid[trm].sum()))
            rows.append(r)
    results["tables"]["search_vs_pmax"] = rows

    # descriptive: by channel
    desc = df.groupby("channel").agg(signups=("signup_id", "size"), paid_rate=("converted_paid_30d", "mean"),
                                      revenue_per_signup=("revenue_180d_eur", "mean")).reset_index()
    results["tables"]["by_channel"] = desc.round(3).to_dict("records")
    # same behaviour, different channel
    beh = df.assign(seg=np.select(
        [df.calendar_connected_72h.eq(1), df.phone_connected_24h.eq(1)],
        ["calendar_72h", "phone_only_24h"], "nothing_24h"))
    results["tables"]["behaviour_by_channel"] = beh.groupby(["seg", "channel"]).agg(
        n=("signup_id", "size"), paid_rate=("converted_paid_30d", "mean"),
        revenue=("revenue_180d_eur", "mean")).round(3).reset_index().to_dict("records")
    # how many payers are already visible at 24h / 72h
    d = df["days_to_first_payment"]
    results["tables"]["paid_visible"] = {"payers": int(y_paid.sum()),
                                         "paid_by_24h": int(d.le(1).sum()), "paid_by_72h": int(d.le(3).sum())}
    results["split"] = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                        "train_period": "2025-10-01..2026-01-31", "test_period": "2026-02-01..2026-03-31"}

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    # scored test set for downstream use
    sc = []
    for g, (score, tem) in main_scores.items():
        sc.append(pd.DataFrame({"signup_id": df.loc[tem, "signup_id"].values, "platform": g,
                                "predicted_value_eur": score.round(2),
                                "actual_revenue_180d": y_rev[tem], "paid_30d": y_paid[tem]}))
    pd.concat(sc).to_csv(OUT / "scored_test_set.csv", index=False)

    rows = [{k: r.get(k) for k in ["name", "n_train", "n", "auc_paid", "revenue_top30_share", "revenue_top50_share", "predicted_mean", "actual_mean"]} for r in results["experiments"]]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for g in ["meta", "google"]:
        print(f"\n== tertiles {g} =="); print(pd.DataFrame(results["tables"][f"tertiles_{g}"]).round(3).to_string(index=False))
        print(f"== calibration {g} =="); print(pd.DataFrame(results["tables"][f"calibration_{g}"]).round(1).to_string(index=False))
        print(f"== top coefficients {g} =="); c = pd.Series(results["tables"][f"coefficients_{g}"]); print(pd.concat([c.head(6), c.tail(8)]).to_string())


if __name__ == "__main__":
    main()
