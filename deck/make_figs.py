import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
GAME, MKT, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{out}/{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


# A: normal map, curves move together
g = [.50, .52, .55, .53, .48, .45, .47, .51, .55, .58, .56, .60, .62, .59, .57, .61, .65, .68, .66, .70, .73, .76, .80, .85]
m = [.47, .50, .54, .55, .50, .44, .45, .49, .53, .56, .57, .58, .60, .60, .58, .59, .63, .66, .67, .69, .71, .75, .78, .83]
r = np.arange(1, len(g) + 1)
fig, ax = plt.subplots(figsize=(8.16, 1.95))
ax.plot(r, g, "-o", ms=2.5, color=GAME, label="game model")
ax.plot(r, m, "-o", ms=2.5, color=MKT, label="Polymarket")
ax.set_ylim(.3, .95); pct(ax); ax.set_xlabel("round"); ax.set_ylabel("P(Team A wins)")
ax.legend(loc="upper left", fontsize=7, frameon=True)
ax.text(24, .34, "max gap 3 pts  ·  no alert", ha="right", fontsize=7, color=GREY)
save(fig, "ex_a_normal")

# B: price moves, game doesn't
g = [.50, .52, .50, .48, .47, .49, .46, .45, .44, .45, .44, .43, .45, .44, .46, .45]
m = [.49, .51, .50, .47, .48, .48, .46, .47, .45, .60, .72, .71, .70, .69, .70, .68]
r = np.arange(1, len(g) + 1)
fig, ax = plt.subplots(figsize=(8.16, 1.95))
gp, mp = np.array(g), np.array(m)
ax.fill_between(r, gp, mp, where=(mp - gp) >= .15, color=RED, alpha=.18, label="gap ≥ 15 pts", interpolate=True)
ax.plot(r, g, "-o", ms=2.5, color=GAME, label="game model")
ax.plot(r, m, "-o", ms=2.5, color=MKT, label="Polymarket")
ax.annotate("+27 pts in 2 min\nno kill, plant or round win", xy=(11, .72), xytext=(12.3, .56),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_ylim(.3, .85); pct(ax); ax.set_xlabel("round"); ax.set_ylabel("P(Team A wins)")
ax.legend(loc="upper left", fontsize=7, frameon=True)
save(fig, "ex_b_alert")

# C: outsized trade
b = [1.8, 2.4, 1.2, 2.0, 2.9, 1.5, 2.2, 1.9, 2.6, 1.4, 2.1, 2.8, 1.7, 2.3, 1.6, 2.0, 2.5, 1.3, 2.2, 1.9, 13.4, 3.1, 2.4, 1.8, 2.7, 2.0, 1.6, 2.3, 2.9, 1.8]
x = np.arange(len(b))
fig, ax = plt.subplots(figsize=(8.16, 1.95))
ax.bar(x, b, color=[MKT if i == 20 else GREY for i in x], width=.8)
ax.axhline(np.median(b), ls="--", lw=.8, color="black", label=f"median fill ({np.median(b):.1f}k shares)")
ax.annotate("6× median, 2 min before the move", xy=(20, 13.4), xytext=(22, 11), fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_xlabel("minute of map"); ax.set_ylabel("fill size (k shares)")
ax.legend(loc="upper left", fontsize=7)
save(fig, "ex_c_trade")

# D: player off own baseline
d = [92, 78, 105, 88, 71, 96, 84, 110, 90, 79, 86, 99, 41, 28, 35, 52, 22, 30, 38, 25]
r = np.arange(1, len(d) + 1)
mu, sd = np.mean(d[:12]), np.std(d[:12])
fig, ax = plt.subplots(figsize=(8.16, 1.95))
ax.axhspan(mu - sd, mu + sd, color=GAME, alpha=.12, label="own first-half range")
ax.axvline(12.5, ls=":", lw=.8, color=GREY)
ax.plot(r, d, "-o", ms=2.5, color="black", label="player P3, damage per round")
ax.plot(r[12:], d[12:], "o", ms=3.5, color=MKT)
ax.text(16.5, 62, "second half z = −2.4", fontsize=7, color=MKT)
ax.set_ylim(0, 130); ax.set_xlabel("round"); ax.set_ylabel("damage")
ax.legend(loc="upper right", fontsize=7)
save(fig, "ex_d_player")

# Coverage: real numbers from the spec (860 Bo3 series, Mar 23 - Apr 20 2026)
labels = ["Polymarket", "OddsPapi", "either source"]
vals = [665, 444, 710]
fig, ax = plt.subplots(figsize=(9.0, 1.85))
ax.barh(labels, [860] * 3, color="#e6e6e6")
bars = ax.barh(labels, vals, color=[MKT, GREY, GAME])
for bar, v in zip(bars, vals):
    ax.text(v + 8, bar.get_y() + bar.get_height() / 2, f"{v} / 860  ({v / 860:.0%})", va="center", fontsize=7.5)
ax.set_xlim(0, 1000); ax.invert_yaxis(); ax.set_xlabel("best-of-3 series with live prices, Mar 23 – Apr 20 2026")
ax.grid(axis="y", visible=False)
save(fig, "coverage")
print("ok")
