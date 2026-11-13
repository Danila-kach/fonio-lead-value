# Fonio case study: what is a signup worth?

Meta and Google Ads optimise for signups, but signups are not equal. This
project builds a model that, 72 hours after signup, predicts how much a
signup is worth in euros, and explains how to send that value back to the
ad platforms, what data setup runs it, and how to prove it works.

## Start here

1. **`docs/writeup.pdf`**: two pages, answers to all four tasks in the order they were asked.
2. **`notebooks/lead_value_model.ipynb`**: Task 1 in full, with the data, the model, every check and every number. Each step is explained in plain language before the code. `lead_value_model.html` is the same notebook as a web page, no setup needed.

## The result in one table

Trained on signups from October 2025 to January 2026, tested on February and March 2026, which the model had not been shown.

| Test signups, February-March 2026 | Meta | Google Ads (Search + PMax) |
|---|---|---|
| Share of all revenue brought by the 30% of signups with the highest predicted value | **85%** | **74%** |
| Same, when every signup counts the same (today) | 30% | 30% |
| Share who paid, among the third of signups with the highest predicted value | 39% | 73% |
| Share who paid, among the third with the lowest predicted value | 2% | 8% |
| Average predicted value / average real revenue per signup | € 93 / € 103 | € 282 / € 289 |

## How the model works, in short

- One model for Meta signups, one for Google Ads signups. Organic and referral signups are not used: there is no platform to send their value to.
- We also tested one model trained on all 6,000 signups. It ranks signups just as well, but without the channel as an input it gets the euro level wrong: it predicts €124 per Meta signup where the truth is €103 (+20%) and €254 per Google signup where the truth is €289 (−12%). The reason is that the same behaviour pays less from Meta: a Meta signup that connected the calendar pays 46% of the time, a Google Search signup 67%. Separate models avoid this.
- Google Search and PMax share one model: they are two campaign types of the same account bidding on the same conversion action, and a PMax-only model (545 training signups) is not better.
- Each model has two steps: **will this signup pay within 30 days?** (a logistic regression: every fact gets a weight, the weights add up to a probability) and **if they pay, how much over 180 days?** (a second formula trained on the signups that did pay). Value = step 1 × step 2.
- It uses only what the database shows 72 hours after signup: the signup form, the device, what the person did in the first three days, and whether they already paid.
- Scored at 72 hours: scoring at 24 hours gives 81% (Meta) and 65% (Google) instead of 85% and 74%.

## What is where

```
docs/writeup.pdf                  the 2-page write-up
notebooks/lead_value_model.ipynb  Task 1 notebook (executed) + .html copy
data/fonio_case_signups.csv       the case dataset
src/features.py                   which facts are known at 24h / 72h, and which columns are excluded
src/model.py                      the two-step model
src/evaluate.py                   the checks: revenue captured by the top 30%, thirds, calibration
src/run_experiments.py            runs every comparison and writes outputs/results.json
src/make_figures.py               draws the charts in outputs/figures/
outputs/results.json              every number used in the write-up and notebook
outputs/scored_test_set.csv       the Feb-Mar signups with the model's value and the real outcome
outputs/figures/                  charts and the Task 3 diagram
requirements.txt                  the Python libraries needed to run the code
```

## Run it yourself

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_experiments.py
.venv/bin/python src/make_figures.py
```

## Tooling

Python 3.11 with pandas (data handling), scikit-learn (logistic regression,
ridge regression, gradient boosting for comparison), matplotlib (charts) and
Jupyter (the notebook). Claude Code was used throughout as a pair programmer
and sparring partner: profiling the data, first drafts of the code and
charts, and challenging the choices (leakage, channel as a feature, separate
vs pooled models, what the platforms' value field accepts). Decisions and the
final text are mine; every number is reproduced by the scripts above.
