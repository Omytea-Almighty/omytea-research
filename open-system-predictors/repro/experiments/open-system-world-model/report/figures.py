"""Report figures from result JSON files (any phase directory). Descriptive only; the verdicts come from osw.analyze.

  python report/figures.py runs/confirm report/fig

Fig. 1  per-horizon KL(p_Bayes || p_model), mean over seeds with min-max band, one panel per task configuration
Fig. 2  H2 sweep: extrapolation-window KL against environment dimension d_E, per-seed points + mean, L as reference
Fig. 3  revival diagnostic D(k) = TV(p_k, p_inf) on k in [51, 200], U2 vs L (simple, tau = 20)

Colour: categorical slots in fixed order per model (validated palette, light surface); identity is also carried by
line style and a direct end label, because three slots sit below 3:1 contrast on the surface.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # model -> (colour, line style); fixed order, never cycled
    "U2": ("#2a78d6", "-"), "L": ("#eb6834", "-"), "C": ("#1baf7a", "--"), "GRU": ("#eda100", "-."), "HMM": ("#e87ba4", ":"),
}
PANELS = [("simple", 20.0), ("metastable", 20.0), ("metastable", 80.0), ("simple", 80.0)]


def load(d: Path):
    R = defaultdict(dict)
    for f in d.glob("*.json"):
        if f.name.startswith("RESULTS"):
            continue
        r = json.loads(f.read_text())
        R[(r["task"], float(r["tau"]), r["model"])][r["seed"]] = r
    return R


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def place_labels(ax, items, min_gap=0.22):
    """Direct end labels without collisions: items = [(x, y, text)] on a log-y axis; nudge in log10 units."""
    items = sorted(items, key=lambda t: t[1])
    ys = [np.log10(y) for _, y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (x, _, t), y in zip(items, ys):
        ax.annotate(t, (x, 10 ** y), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK, va="center")


def fig_kl(R, out):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 3.4), sharey=True, facecolor=SURFACE)
    for ax, (task, tau) in zip(axes, PANELS):
        style(ax)
        ends = []
        for m, (c, ls) in SERIES.items():
            runs = R.get((task, tau, m))
            if not runs:
                continue
            Y = np.array([r["per_k"]["kl"] for r in runs.values()])
            k = np.arange(1, Y.shape[1] + 1)
            Y = np.clip(Y, 1e-6, None)
            ax.fill_between(k, Y.min(0), Y.max(0), color=c, alpha=0.12, linewidth=0)
            ax.plot(k, Y.mean(0), color=c, ls=ls, lw=2)
            ends.append((k[-1], Y.mean(0)[-1], m))
        place_labels(ax, ends)
        ax.axvline(50, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_yscale("log"); ax.set_xlim(1, 215)
        ax.set_title(f"{task}, τ{'_slow' if task == 'metastable' else ''} = {tau:g}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("horizon k", fontsize=8, color=MUTED)
    axes[0].set_ylabel("KL(p_Bayes ‖ p_model), nats", fontsize=8, color=MUTED)
    axes[0].text(52, axes[0].get_ylim()[1] * 0.5, "extrapolation →", fontsize=7, color=MUTED)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, label=m) for m, (c, ls) in SERIES.items()]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle("Per-horizon KL to the exact Bayes predictor (mean over seeds, band = min–max)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig1-kl-per-horizon.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_sweep(R, out):
    dims = {"U1": 1, "U2": 2, "U4": 4, "U8": 8}
    fig, ax = plt.subplots(figsize=(4.6, 3.4), facecolor=SURFACE)
    style(ax)
    means = []
    for m, d in dims.items():
        runs = R.get(("simple", 20.0, m), {})
        v = [r["windows"]["extrap_51_200"]["kl"] for r in runs.values()]
        if v:
            ax.scatter(np.full(len(v), d), v, s=18, color=SERIES["U2"][0], alpha=0.45, linewidths=0)
            means.append((d, float(np.mean(v))))
    if means:
        ax.plot(*zip(*means), color=SERIES["U2"][0], lw=2, marker="o", ms=6)
    L = [r["windows"]["extrap_51_200"]["kl"] for r in R.get(("simple", 20.0, "L"), {}).values()]
    if L:
        ax.axhline(np.mean(L), color=SERIES["L"][0], lw=2, ls="-")
        ax.annotate("L (mean)", (8, np.mean(L)), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yscale("log")
    ax.set_xlabel("environment dimension d_E of the unitary predictor", fontsize=8, color=MUTED)
    ax.set_ylabel("extrapolation KL, k ∈ [51, 200]", fontsize=8, color=MUTED)
    ax.set_title("H2 sweep (simple, τ = 20); dots = seeds", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig2-h2-sweep.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_revival(R, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.2), facecolor=SURFACE)
    style(ax)
    for m in ("U2", "L"):
        runs = R.get(("simple", 20.0, m), {})
        curves = [r["revival"]["D_curve"] for r in runs.values() if "D_curve" in r.get("revival", {})]
        if not curves:
            continue
        Y = np.array(curves); k = np.arange(51, 51 + Y.shape[1])
        c, ls = SERIES[m]
        ax.plot(k, Y.mean(0), color=c, ls=ls, lw=2)
        ax.annotate(m, (k[-1], Y.mean(0)[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.set_xlabel("horizon k", fontsize=8, color=MUTED)
    ax.set_ylabel("D(k) = TV(p_k, p_∞)", fontsize=8, color=MUTED)
    ax.set_title("Revival diagnostic, simple τ = 20 (mean over sequences and seeds)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig3-revival.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/confirm")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "report/fig")
    out.mkdir(parents=True, exist_ok=True)
    R = load(src)
    fig_kl(R, out); fig_sweep(R, out); fig_revival(R, out)
    print(f"figures written to {out}")
