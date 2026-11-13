# Fonio case study: what is a signup worth?

**Interactive dashboard: https://danila-kach.github.io/fonio-lead-value/**

Meta and Google Ads optimise for signups, but signups are not equal. This
project builds a model that, 72 hours after signup, predicts how much a
signup is worth in euros, and explains how to send that value back to the
ad platforms, what data setup runs it, and how to prove it works.

## Start here

1. **`docs/writeup.pdf`**: two pages, answers to all four tasks in the order they were asked.
2. **Interactive dashboard**: https://danila-kach.github.io/fonio-lead-value/ (switch Meta / Google Ads, drag the slider, try the value calculator, browse every test signup).
3. **`notebooks/lead_value_model.ipynb`**: Task 1 in full. The data, the features and what is left out, the model, the checks and every number, each step explained in plain language before the code. Self-contained: it needs only pandas, numpy, scikit-learn and matplotlib and loads the CSV from this repository, so it runs on Google Colab as is (File → Open notebook → GitHub → paste this repository's URL). GitHub also shows it with all tables and charts.

## The result in one table

Trained on signups from October 2025 to January 2026, tested on February and March 2026, which the models had not been shown.

| Test signups, February-March 2026 | Meta | Google Ads (Search + PMax) |
|---|---|---|
| Real revenue per signup, third with the highest predicted value | € 270 | € 666 |
| Real revenue per signup, third with the lowest predicted value | € 7 | € 42 |
| Real revenue per signup, every signup the same (today) | € 103 | € 289 |
| AUC, "will they pay?" | 0.87 | 0.86 |
| R² of the value on groups of ~80 signups | 0.96 | 0.99 |

## How the model works, in short

- One model for Meta signups, one for Google Ads signups. Organic and referral signups are not used: there is no platform to send their value to.
- We also tested one model trained on all 6,000 signups. It ranks signups just as well, but without the channel as an input it gets the euro level wrong: € 124 per Meta signup where the truth is € 103 (+20%). The same behaviour pays less from Meta: a Meta signup that connected the calendar pays 46% of the time, a Google Search signup 67%. Separate models avoid this.
- Google Search and PMax share one model: two campaign types of the same account bidding on the same conversion action; a PMax-only model (545 training signups) is not better.
- Each model has two steps: **will this signup pay within 30 days?** (a logistic regression: every fact gets a weight, the weights add up to a probability) and **if they pay, how much over 180 days?** (a second formula trained on the signups that did pay). Value = step 1 × step 2.
- It uses only what the database shows 72 hours after signup: the signup form, the device, what the person did in the first three days, and whether they already paid.
- Scored at 72 hours: scoring at 24 hours is less accurate (the calendar, the test calls and early payments are not visible yet).

## What is where

```
docs/writeup.pdf                  the 2-page write-up
docs/index.html + docs/data.js    the interactive dashboard (served by GitHub Pages from this folder)
notebooks/lead_value_model.ipynb  Task 1 notebook, executed, self-contained
scripts/build_dashboard_data.py   rebuilds docs/data.js (trains the same two models, scores the test signups)
data/fonio_case_signups.csv       the case dataset (synthetic)
requirements.txt                  the Python libraries used
```

## Run it yourself

Open the notebook on Google Colab, or locally:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook notebooks/lead_value_model.ipynb
.venv/bin/python scripts/build_dashboard_data.py
```

## Tooling

Python 3.11 with pandas (data handling), scikit-learn (logistic regression,
ridge regression, gradient boosting for comparison), matplotlib (charts) and
Jupyter (the notebook). Claude Code was used throughout as a pair programmer
and sparring partner: profiling the data, first drafts of the code and
charts, and challenging the choices (leakage, channel as a feature, separate
vs pooled models, what the platforms' value field accepts). Decisions and the
final text are mine; every number is reproduced by the notebook.
