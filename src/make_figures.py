"""Charts for the write-up, built from outputs/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "outputs" / "results.json").read_text())
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892", "#0b0b0b", "#52514e"
SEQ = ["#c7dcf5", "#6ea6e6", "#2a78d6"]  # light -> dark blue
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
EXP = {e["name"]: e for e in R["experiments"]}
PLATFORMS = [("meta", "Meta"), ("google", "Google Ads (Search + PMax)")]


def gain_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (g, title) in zip(axes, PLATFORMS):
        gc = R["tables"][f"gain_{g}"]
        x = [0] + gc["share"]; y = [0] + gc["captured"]
        ax.plot([0, 1], [0, 1], color=GRAY, lw=2, ls="--", label="Today: every signup counts the same")
        ax.plot(x, y, color=BLUE, lw=2, label="Lead-value model (72h)")
        ax.scatter([0.3], [EXP[f"{g}_72h_main"]["revenue_top30_share"]], color=BLUE, s=40, zorder=3)
        v = EXP[f"{g}_72h_main"]["revenue_top30_share"]
        ax.annotate(f"top 30% of signups\n= {v:.0%} of revenue", (0.3, v), xytext=(0.36, v - 0.25), color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Share of signups taken, highest predicted value first")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color="#eeede8")
    axes[0].set_ylabel("Share of all revenue_180d_eur they brought")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Test months Feb-Mar 2026: share of revenue brought by the signups with the highest predicted value",
                 x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "gain_curves.png", dpi=200); plt.close(fig)


def tertiles():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (g, title) in zip(axes, PLATFORMS):
        t = pd.DataFrame(R["tables"][f"tertiles_{g}"])
        x = np.arange(3)
        bars = ax.bar(x, t["revenue_per_lead"], color=SEQ[::-1], width=0.6)
        for b, v, p in zip(bars, t["revenue_per_lead"], t["paid_rate"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 10, f"€{v:.0f}\n{p:.0%} pay", ha="center", va="bottom", color=INK, fontsize=9)
        avg = EXP[f"{g}_72h_main"]["actual_mean"]
        ax.axhline(avg, color=GRAY, ls="--", lw=1.2); ax.text(2.45, avg, f"avg €{avg:.0f}", va="bottom", ha="right", color=INK2, fontsize=8)
        ax.set_xticks(x, ["Top third", "Middle third", "Bottom third"])
        ax.set_ylim(0, t["revenue_per_lead"].max() * 1.35)
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#eeede8")
    axes[0].set_ylabel("Average real revenue_180d_eur per signup (€)")
    fig.suptitle("Test signups (Feb-Mar 2026) split into thirds by predicted value", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "tertiles.png", dpi=200); plt.close(fig)


def calibration():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (g, title) in zip(axes, PLATFORMS):
        c = pd.DataFrame(R["tables"][f"calibration_{g}"])
        m = max(c["predicted"].max(), c["actual"].max()) * 1.1
        ax.plot([0, m], [0, m], color=GRAY, ls="--", lw=1.2, label="Perfect calibration")
        ax.plot(c["predicted"], c["actual"], color=BLUE, lw=2, marker="o", ms=6, label="Model, by quintile")
        ax.set_xlabel("Average predicted value per signup (€)"); ax.set_title(title, color=INK, loc="left", fontweight="bold")
        ax.set_xlim(0, m); ax.set_ylim(0, m); ax.grid(color="#eeede8")
    axes[0].set_ylabel("Average real revenue_180d_eur per signup (€)"); axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Does €300 predicted mean €300 earned? Test signups in 5 groups by predicted value", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=200); plt.close(fig)


def horizon():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = ["Today\n(every signup the same)", "Model, scored at 24h", "Model, scored at 72h"]
    keys = ["all_equal", "24h", "72h_main"]
    x = np.arange(len(labels)); w = 0.36
    for i, (g, title) in enumerate(PLATFORMS):
        vals = [EXP[f"{g}_{k}"]["revenue_top30_share"] for k in keys]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=[BLUE, ORANGE][i], label=title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Revenue brought by the top 30% of signups"); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", color="#eeede8")
    ax.set_title("Speed vs accuracy: revenue brought by the 30% of signups with the highest predicted value", loc="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "horizon_tradeoff.png", dpi=200); plt.close(fig)


def behaviour_by_channel():
    b = pd.DataFrame(R["tables"]["behaviour_by_channel"])
    order = ["meta", "google_pmax", "organic", "referral", "google_search"]
    segs = [("nothing_24h", "Did nothing in 24h"), ("phone_only_24h", "Phone connected, no calendar"), ("calendar_72h", "Calendar connected by 72h")]
    fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(order)); w = 0.26
    for i, (s, lab) in enumerate(segs):
        vals = [b[(b.seg == s) & (b.channel == c)]["paid_rate"].iloc[0] for c in order]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=SEQ[i], label=lab)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, order); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_ylim(0, 0.85)
    ax.set_ylabel("Paid within 30 days"); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", color="#eeede8")
    ax.set_title("Same onboarding behaviour, different channel: Meta signups pay less at every level", loc="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "behaviour_by_channel.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    gain_curves(); tertiles(); calibration(); horizon(); behaviour_by_channel()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


def value_distribution(platform="meta", label="Meta"):
    """Every test signup sorted by the value we send to the platform, thirds coloured; next to it: today."""
    sc = pd.read_csv(ROOT / "outputs" / "scored_test_set.csv")
    v = np.sort(sc.loc[sc.platform == platform, "predicted_value_eur"].values)
    n = len(v); x = np.arange(1, n + 1)
    prof = {r["third"]: r for r in R["tables"][f"third_profile_{platform}"]}
    today = EXP[f"{platform}_72h_main"]["actual_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.5]})
    ymax = max(v.max() * 1.05, 400)
    # left: today
    ax = axes[0]
    ax.fill_between(x, 0, today, color="#d6d5d0"); ax.plot(x, np.full(n, today), color=GRAY, lw=2)
    ax.text(n / 2, today + ymax * 0.03, f"every signup counts the same\n(the average signup brings € {today:.0f})", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_title(f"Today: what {label} is told", color=INK, loc="left", fontweight="bold")
    ax.set_xlabel(f"{n} test signups"); ax.set_ylabel("Value of a signup (€)"); ax.set_xlim(0, n); ax.set_ylim(0, ymax); ax.grid(axis="y", color="#eeede8")
    # right: model
    ax = axes[1]
    b1, b2 = n // 3, 2 * n // 3
    for lo, hi, col, key in [(0, b1, SEQ[0], "bottom"), (b1, b2, SEQ[1], "middle"), (b2, n, SEQ[2], "top")]:
        ax.fill_between(x[lo:hi], 0, v[lo:hi], color=col, step=None)
    ax.plot(x, v, color=BLUE, lw=1.5)
    for xx in (b1, b2): ax.axvline(xx, color=INK2, lw=0.8, ls=":")
    p = prof["bottom"]; ax.text(b1 / 2, ymax * 0.30, f"Bottom third\nsent: € {p['value_sent']:.0f} per signup\nreally brought: € {p['revenue_180d']:.0f}\n\n{p['phone_connected_24h']:.0%} connected the phone\n{p['calendar_connected_72h']:.0%} connected the calendar\n{p['solo_company']:.0%} one-person companies", ha="center", va="center", fontsize=8, color=INK)
    p = prof["middle"]; ax.text((b1 + b2) / 2, ymax * 0.30, f"Middle third\nsent: € {p['value_sent']:.0f} per signup\nreally brought: € {p['revenue_180d']:.0f}\n\n{p['phone_connected_24h']:.0%} connected the phone\n{p['calendar_connected_72h']:.0%} connected the calendar\n{p['test_calls_72h']:.1f} test calls on average", ha="center", va="center", fontsize=8, color=INK)
    p = prof["top"]; ax.text((b2 + n) / 2 - n * 0.02, ymax * 0.72, f"Top third\nsent: € {p['value_sent']:.0f} per signup\nreally brought: € {p['revenue_180d']:.0f}\n\n{p['phone_connected_24h']:.0%} connected the phone\n{p['calendar_connected_72h']:.0%} connected the calendar\n{p['test_calls_72h']:.1f} test calls on average\n{p['business_email']:.0%} business email", ha="center", va="center", fontsize=8, color=INK,
                              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d6d5d0"))
    ax.set_title(f"With the model: what {label} is told", color=INK, loc="left", fontweight="bold")
    ax.set_xlabel(f"{n} test signups, sorted by the value the model sends"); ax.set_xlim(0, n); ax.grid(axis="y", color="#eeede8")
    fig.suptitle(f"{label}, test signups February-March 2026: the value {label} receives per signup, today vs with the model", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"value_distribution_{platform}.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    value_distribution("meta", "Meta"); value_distribution("google", "Google Ads")
    print("value distribution figures saved")
