"""Figures for the report: results/*.csv -> figures/*.png

Owned by: Somyak Mohanta (241CS257)

    python -m src.plots

Every figure is generated from a committed CSV, so the report can be rebuilt without
re-running the sweeps, and the numbers in the text always match the pictures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as mticker      # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# One colour and one marker per policy, used consistently across every figure so the
# reader learns the legend once. Okabe-Ito hues plus a purple for FCFS (black reads as
# ink, not as a series). Checked with a CVD validator: every pair separates for normal
# vision; the closest pair under deuteranopia (FCFS / SJF-Oracle) is also told apart by
# marker shape, which is why every policy has its own marker.
COLORS = {
    "FCFS": "#6A3D9A",
    "SJF-Oracle": "#0072B2",
    "SJF-Pred": "#56B4E9",
    "SRTF": "#009E73",
    "MLFQ-LAS": "#D55E00",
    "MLFQ-Pred": "#E69F00",
    "Priority-SLO": "#CC79A7",
}
MARKERS = {
    "FCFS": "o", "SJF-Oracle": "s", "SJF-Pred": "^", "SRTF": "v",
    "MLFQ-LAS": "D", "MLFQ-Pred": "X", "Priority-SLO": "P",
}
ORDER = list(COLORS)
MODES = ("recompute", "swap")
MODE_TITLE = {"recompute": "recompute preemption (vLLM default)",
              "swap": "swap preemption (FastServe-style)"}
INK, MUTED = "#222222", "#777777"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
})


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _log(ax, axis="y"):
    """Log scale with plain-decimal ticks at 1-2-5 steps: '0.05', not '5x10^-2'."""
    getattr(ax, f"set_{axis}scale")("log")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    target.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(mticker.NullFormatter())


def _policies_in(df):
    return [p for p in ORDER if p in set(df["policy"])]


def _line(ax, x, y, pol, **kw):
    ax.plot(x, y, marker=MARKERS[pol], color=COLORS[pol], label=pol,
            markeredgecolor="white", markeredgewidth=0.6, **kw)


def _legend_on_top(fig, axes, ncol=7):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def fig_capacity():
    """Where the engine saturates. Everything else is read relative to this knee."""
    df = pd.read_csv(RESULTS / "e0_capacity.csv").sort_values("arrival_rate")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    peak = df["throughput_req_s"].max()

    ax1.plot(df["arrival_rate"], df["throughput_req_s"], "o-", color=COLORS["FCFS"],
             label="completed")
    ax1.plot([0, df["arrival_rate"].max()], [0, df["arrival_rate"].max()],
             "--", color=MUTED, lw=1, label="offered load (y = x)")
    ax1.axhline(peak, color=MUTED, ls=":", lw=1.2, label=f"capacity = {peak:.2f} req/s")
    ax1.legend(fontsize=8)
    _style(ax1, "Engine capacity (FCFS)", "offered load (requests/s)", "completed (requests/s)")

    ax2.plot(df["arrival_rate"], df["ttft_p99"], "o-", color=COLORS["FCFS"])
    ax2.axvline(peak, color=MUTED, ls=":", lw=1.2)
    _log(ax2)
    ax2.annotate("queue grows without\nbound past capacity",
                 xy=(peak, df["ttft_p99"].median()),
                 xytext=(peak + 3.5, df["ttft_p99"].median() / 8), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    _style(ax2, "P99 time to first token", "offered load (requests/s)", "TTFT p99 (s, log)")

    fig.suptitle(f"E0  The modelled engine saturates at {peak:.2f} req/s; "
                 "scheduling only matters above it", fontsize=11, y=1.04)
    return _save(fig, "fig1_capacity.png")


def fig_load_sweep():
    """E1: latency vs load for all seven policies, under both preemption mechanisms."""
    df = pd.read_csv(RESULTS / "e1_load_sweep.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey="col")

    for row, mode in enumerate(MODES):
        dm = df[df["preemption_mode"] == mode]
        for pol in _policies_in(dm):
            d = dm[dm["policy"] == pol].sort_values("arrival_rate")
            _line(axes[row, 0], d["arrival_rate"], d["norm_latency_mean"], pol)
            _line(axes[row, 1], d["arrival_rate"], d["max_wait"], pol)
        _log(axes[row, 0])
        _style(axes[row, 0], f"Mean normalized latency -- {mode}",
               "offered load (requests/s)" if row else "", "s per output token (log)")
        _style(axes[row, 1], f"Worst total wait -- {mode}",
               "offered load (requests/s)" if row else "", "max wait (s)")

    _legend_on_top(fig, axes[0])
    fig.suptitle("E1  Below saturation nothing differs. Above it, the cost of preemption "
                 "decides which policies work", fontsize=11, y=1.04)
    return _save(fig, "fig2_load_sweep.png")


def fig_tradeoff():
    """The Pareto picture: average latency against worst-case wait, at 10 req/s."""
    df = pd.read_csv(RESULTS / "e1_load_sweep.csv")
    df = df[df["arrival_rate"] == 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)

    for ax, mode in zip(axes, MODES):
        dm = df[df["preemption_mode"] == mode]
        for pol in _policies_in(dm):
            d = dm[dm["policy"] == pol]
            ax.scatter(d["norm_latency_mean"], d["max_wait"], s=70, color=COLORS[pol],
                       marker=MARKERS[pol], label=pol, zorder=3,
                       edgecolor="white", linewidth=1)
        _log(ax, "x")
        ax.annotate("better", xy=(0.07, 0.07), xycoords="axes fraction", fontsize=9,
                    color=MUTED, ha="center")
        ax.annotate("", xy=(0.02, 0.02), xytext=(0.15, 0.15), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
        _style(ax, MODE_TITLE[mode], "mean normalized latency (s/token, log)",
               "max wait (s)" if mode == "recompute" else "")

    _legend_on_top(fig, axes)
    fig.suptitle("E1  The trade-off at 10 req/s: no policy is best on both axes "
                 "(SJF-Pred sits under SJF-Oracle: sigma = 0)", fontsize=11, y=1.06)
    return _save(fig, "fig3_tradeoff.png")


def fig_prediction_error():
    """E3: how good does a length predictor have to be?"""
    df = pd.read_csv(RESULTS / "e3_prediction_error.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, mode in zip(axes, MODES):
        dm = df[df["preemption_mode"] == mode]
        for pol in ("SJF-Pred", "SRTF", "MLFQ-Pred"):
            d = dm[dm["policy"] == pol].sort_values("predictor_rank_corr")
            _line(ax, d["predictor_rank_corr"], d["norm_latency_mean"], pol)
        # Policies that never read the prediction are flat by construction.
        label_pos = ax.get_yaxis_transform()
        for pol, txt, above in (("FCFS", "FCFS", True), ("SJF-Oracle", "SJF-Oracle", True),
                                ("MLFQ-LAS", "MLFQ-LAS (reads no prediction)", False)):
            y = dm[dm["policy"] == pol]["norm_latency_mean"].iloc[0]
            ax.axhline(y, color=COLORS[pol], ls="--", lw=1.2)
            ax.text(0.98, y * (1.05 if above else 0.95), txt, fontsize=8, color=INK,
                    ha="right", va="bottom" if above else "top", transform=label_pos)
        _log(ax)
        ax.invert_xaxis()
        _style(ax, MODE_TITLE[mode], "predictor quality (Spearman rank corr., perfect -> useless)",
               "mean normalized latency (s/token, log)" if mode == "recompute" else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("E3  A mediocre length predictor captures most of the benefit",
                 fontsize=11, y=1.06)
    return _save(fig, "fig4_prediction_error.png")


def fig_starvation_valve():
    """E4: MLFQ's ageing threshold as an explicit latency/fairness dial (swap preemption)."""
    df = pd.read_csv(RESULTS / "e4_starvation_valve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for pol in ("MLFQ-LAS", "MLFQ-Pred"):
        d = df[df["policy"] == pol].sort_values("starvation_s")
        labels = [f"{v:g}" if v < 1e8 else "off" for v in d["starvation_s"]]
        x = np.arange(len(labels))
        _line(axes[0], x, d["norm_latency_mean"], pol)
        _line(axes[1], x, d["max_wait"], pol)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        label_pos = ax.get_yaxis_transform()
        col = "norm_latency_mean" if ax is axes[0] else "max_wait"
        for ref in ("FCFS", "SJF-Oracle"):
            y = df[df["policy"] == ref][col].iloc[0]
            ax.axhline(y, color=COLORS[ref], ls="--", lw=1.2)
            ax.text(0.02, y * 1.04, ref, fontsize=8, color=INK, transform=label_pos)
    _log(axes[0])
    _style(axes[0], "Mean normalized latency", "ageing threshold (s)", "s per output token (log)")
    _style(axes[1], "Worst total wait", "ageing threshold (s)", "max wait (s)")

    _legend_on_top(fig, axes, ncol=2)
    fig.suptitle("E4  MLFQ's ageing threshold is a dial between FCFS and SJF "
                 "(swap preemption)", fontsize=11, y=1.06)
    return _save(fig, "fig5_starvation_valve.png")


def fig_chunked_prefill():
    """E5: chunk size as the TTFT-versus-TPOT dial, on long-prompt traffic."""
    df = pd.read_csv(RESULTS / "e5_chunked_prefill.csv")
    order = ["off", "1024", "512", "256", "128", "64"]
    df = df[df["chunk_label"].astype(str).isin(order)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    for pol in _policies_in(df):
        d = df[df["policy"] == pol].copy()
        d["ord"] = d["chunk_label"].astype(str).map({k: i for i, k in enumerate(order)})
        d = d.sort_values("ord")
        _line(ax1, d["ord"], d["tpot_mean"], pol)
        _line(ax2, d["ord"], d["throughput_tok_s"], pol)

    for ax, ylab, title in ((ax1, "mean TPOT (s/token, log)", "Streaming smoothness"),
                            (ax2, "throughput (tokens/s)", "Throughput")):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        _style(ax, title, "prefill chunk size (tokens)", ylab)
    _log(ax1)   # MLFQ-LAS thrashing under recompute would otherwise flatten every other line

    _legend_on_top(fig, (ax1, ax2))
    fig.suptitle("E5  Chunked prefill on long prompts (bimodal workload, recompute)",
                 fontsize=11, y=1.06)
    return _save(fig, "fig6_chunked_prefill.png")


def fig_workload_mix():
    """E2: does the ranking survive a change of workload? (vLLM-default recompute)"""
    df = pd.read_csv(RESULTS / "e2_workload_mix.csv")
    df = df[df["preemption_mode"] == "recompute"]
    pols = _policies_in(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    width = 0.38

    for ax, col, title in ((ax1, "norm_latency_mean", "Mean normalized latency (log)"),
                           (ax2, "max_wait", "Worst total wait (s)")):
        for i, wl in enumerate(("default", "bimodal")):
            vals = [df[(df["policy"] == p) & (df["workload"] == wl)][col].iloc[0]
                    for p in pols]
            ax.bar([j + i * width for j in range(len(pols))], vals, width * 0.92,
                   label=wl, color=["#0072B2", "#E69F00"][i], edgecolor="white", linewidth=1)
        ax.set_xticks([j + width / 2 for j in range(len(pols))])
        ax.set_xticklabels(pols, rotation=25, ha="right", fontsize=8)
        _style(ax, title, "", "")
    _log(ax1)
    ax1.legend(fontsize=8, title="workload")

    fig.suptitle("E2  Mixing long batch jobs with short chats widens the gaps "
                 "(recompute preemption)", fontsize=11, y=1.04)
    return _save(fig, "fig7_workload_mix.png")


def fig_preemption_cost():
    """E6: how cheap must preemption be before preemptive policies pay off?"""
    df = pd.read_csv(RESULTS / "e6_preemption_cost.csv")
    # On the T4, recompute costs 35-80x a swap round trip depending on how much KV the
    # victim holds, so recompute sits between x10 and x100 on this cost axis.
    order = ["swap x0.1", "swap x1", "swap x10", "recompute", "swap x100"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for pol in _policies_in(df):
        d = df[df["policy"] == pol].copy()
        d["ord"] = d["cost_label"].map({k: i for i, k in enumerate(order)})
        d = d.sort_values("ord")
        _line(ax1, d["ord"], d["norm_latency_mean"], pol)
        _line(ax2, d["ord"], d["throughput_tok_s"], pol)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o if o != "recompute" else "recompute\n(~x50)" for o in order])
    _log(ax1)
    _style(ax1, "Mean normalized latency", "cost of preempting a request  (cheap -> expensive)",
           "s per output token (log)")
    _style(ax2, "Throughput", "cost of preempting a request  (cheap -> expensive)",
           "tokens/s")

    _legend_on_top(fig, (ax1, ax2), ncol=5)
    fig.suptitle("E6  Preemptive policies win while preemption is cheap and collapse "
                 "under recompute", fontsize=11, y=1.06)
    return _save(fig, "fig8_preemption_cost.png")


def fig_robustness():
    """E7: the headline ratios over 10 seeds and three trace lengths, with 95% CIs."""
    df = pd.read_csv(RESULTS / "e7_summary.csv")
    df = df[(df["metric"] == "norm_latency_mean") & (df["policy"] != "FCFS")]
    sizes = sorted(df["n_requests"].unique())
    pols = [p for p in ORDER if p in set(df["policy"])]
    offsets = dict(zip(sizes, np.linspace(-0.22, 0.22, len(sizes))))
    shades = dict(zip(sizes, ("#9BB8D3", "#0072B2", "#08306B")))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, mode in zip(axes, MODES):
        dm = df[df["preemption_mode"] == mode]
        for size in sizes:
            ds = dm[dm["n_requests"] == size].set_index("policy").loc[pols]
            y = np.arange(len(pols)) + offsets[size]
            ax.errorbar(ds["ratio_vs_fcfs"], y, xerr=ds["ratio_ci95"], fmt="o",
                        color=shades[size], ms=5, capsize=2.5, lw=1.4,
                        label=f"{size} requests")
        ax.axvline(1.0, color=COLORS["FCFS"], ls="--", lw=1.2)
        ax.text(1.0, 0.01, " FCFS = 1", color=INK, fontsize=8, va="bottom",
                transform=ax.get_xaxis_transform())
        _log(ax, "x")
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels(pols)
        ax.invert_yaxis()
        _style(ax, MODE_TITLE[mode], "mean normalized latency / FCFS on the same trace (log)", "")

    _legend_on_top(fig, axes, ncol=3)
    fig.suptitle("E7  Ratios to FCFS over 10 seeds: mean and 95% CI, at three burst lengths",
                 fontsize=11, y=1.06)
    return _save(fig, "fig9_robustness.png")


FIGURE_FNS = [fig_capacity, fig_load_sweep, fig_tradeoff, fig_prediction_error,
              fig_starvation_valve, fig_chunked_prefill, fig_workload_mix,
              fig_preemption_cost, fig_robustness]


def main() -> None:
    if not RESULTS.exists() or not list(RESULTS.glob("*.csv")):
        raise SystemExit("no results found -- run `python -m src.runner --all` first")

    print(f"reading {RESULTS}")
    for fn in FIGURE_FNS:
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  skipped {fn.__name__}: {exc.filename} not found")
    print(f"\nfigures in {FIGURES}")


if __name__ == "__main__":
    main()
