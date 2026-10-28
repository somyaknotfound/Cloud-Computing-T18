"""Every number report.md quotes, recomputed from results/*.csv.

    python report/report_numbers.py

The report is written by hand, so this is how it stays honest: each section below matches
a section of the report, and every figure in the prose should appear here. After the cost
model is recalibrated and the experiments re-run, run this again and update the report
from its output -- never from memory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
POLICIES = ["FCFS", "SJF-Oracle", "SJF-Pred", "SRTF", "MLFQ-LAS", "MLFQ-Pred", "Priority-SLO"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


def head(title: str) -> None:
    print(f"\n## {title}")


def benefit(x: float, fcfs: float, oracle: float) -> float:
    """Share of SJF-Oracle's improvement over FCFS that `x` achieves."""
    return (fcfs - x) / (fcfs - oracle)


def capacity() -> None:
    head("E0 capacity (FCFS)")
    e0 = load("e0_capacity")
    top = e0.loc[e0["throughput_req_s"].idxmax()]
    print(f"capacity {top['throughput_req_s']:.2f} req/s = {top['throughput_tok_s']:.0f} tok/s")
    below = e0[e0["arrival_rate"] < top["throughput_req_s"]]["arrival_rate"].max()
    for rate in (below, e0["arrival_rate"].max()):
        r = e0[e0["arrival_rate"] == rate].iloc[0]
        print(f"TTFT p99 at {rate:g} req/s: {r['ttft_p99']:.2f} s")
    print(f"10 req/s is {10 / top['throughput_req_s']:.1f}x capacity")


def headline() -> None:
    e1 = load("e1_load_sweep")
    for mode in ("recompute", "swap"):
        head(f"E1 at 10 req/s, seed 42, {mode}")
        d = e1[(e1["arrival_rate"] == 10.0) & (e1["preemption_mode"] == mode)].set_index("policy")
        print("| Policy | Mean NL | TTFT p50 | Max wait | Goodput | Jain | Throughput | Preemptions |")
        print("|---|---|---|---|---|---|---|---|")
        for p in POLICIES:
            r = d.loc[p]
            print(f"| {p} | {r['norm_latency_mean']:.3f} | {r['ttft_p50']:.2f} s | "
                  f"{r['max_wait']:.1f} s | {100*r['goodput_frac']:.1f}% | "
                  f"{r['jain_fairness']:.3f} | {r['throughput_tok_s']:.0f} tok/s | "
                  f"{r['preemptions']:.0f} |")
        f, s = d.loc["FCFS"], d.loc["SJF-Oracle"]
        print(f"SJF vs FCFS: NL {f['norm_latency_mean']/s['norm_latency_mean']:.1f}x better, "
              f"TTFT p50 {f['ttft_p50']/s['ttft_p50']:.0f}x better, "
              f"max wait {s['max_wait']/f['max_wait']:.1f}x worse")
        for p in ("SRTF", "MLFQ-LAS", "MLFQ-Pred"):
            r = d.loc[p]
            print(f"{p}: {benefit(r['norm_latency_mean'], f['norm_latency_mean'], s['norm_latency_mean']):.0%}"
                  f" of oracle benefit, max wait {r['max_wait']/f['max_wait']:.2f}x FCFS, "
                  f"throughput {r['throughput_tok_s']/f['throughput_tok_s']:.2f}x FCFS")


def robustness() -> None:
    summ = load("e7_summary")
    raw = load("e7_robustness")
    for mode in ("recompute", "swap"):
        head(f"E7 over 10 seeds, {mode}: ratio to FCFS on the same trace (mean +/- 95% CI)")
        print("| Policy | N | NL ratio | TTFT p50 ratio | Max-wait ratio | Throughput ratio | NL better than FCFS |")
        print("|---|---|---|---|---|---|---|")
        d = summ[summ["preemption_mode"] == mode]
        for p in POLICIES[1:]:
            for n in sorted(d["n_requests"].unique()):
                g = d[(d["policy"] == p) & (d["n_requests"] == n)].set_index("metric")
                cell = lambda m: f"{g.loc[m, 'ratio_vs_fcfs']:.2f} +/- {g.loc[m, 'ratio_ci95']:.2f}"
                print(f"| {p} | {n} | {cell('norm_latency_mean')} | {cell('ttft_p50')} | "
                      f"{cell('max_wait')} | {cell('throughput_tok_s')} | "
                      f"{g.loc['norm_latency_mean', 'seeds_better_than_fcfs']}/10 |")
    head("E7 paired comparisons, N=600")
    r = raw[raw["n_requests"] == 600]
    for mode in ("recompute", "swap"):
        piv = r[r["preemption_mode"] == mode].pivot(index="seed", columns="policy",
                                                    values="norm_latency_mean")
        wins = (piv["SRTF"] < piv["SJF-Oracle"]).sum()
        print(f"{mode}: SRTF beats SJF-Oracle on mean NL in {wins}/10 seeds")
        spread = r[(r["preemption_mode"] == mode)
                   & r["policy"].isin(["FCFS", "SJF-Oracle", "SJF-Pred", "Priority-SLO"])]
        t = spread.groupby("seed")["throughput_tok_s"].agg(lambda s: (s.max() - s.min()) / s.max())
        print(f"{mode}: throughput spread across non-preemptive policies: "
              f"median {t.median():.1%}, max {t.max():.1%}")


def prediction() -> None:
    e3 = load("e3_prediction_error")
    for mode in ("recompute", "swap"):
        head(f"E3 predictor quality, {mode}")
        d = e3[e3["preemption_mode"] == mode]
        fcfs = d[d["policy"] == "FCFS"]["norm_latency_mean"].iloc[0]
        oracle = d[d["policy"] == "SJF-Oracle"]["norm_latency_mean"].iloc[0]
        mlfq = d[d["policy"] == "MLFQ-LAS"]["norm_latency_mean"]
        print(f"FCFS {fcfs:.3f}  SJF-Oracle {oracle:.3f}  "
              f"MLFQ-LAS {mlfq.iloc[0]:.3f} (identical at every sigma: {mlfq.nunique() == 1})")
        print("| sigma | rank corr | SJF-Pred NL | benefit | SRTF NL | MLFQ-Pred NL |")
        print("|---|---|---|---|---|---|")
        piv = d.pivot(index="predictor_error", columns="policy", values="norm_latency_mean")
        rc = d.groupby("predictor_error")["predictor_rank_corr"].first()
        for s in piv.index:
            print(f"| {s:g} | {rc[s]:.2f} | {piv.loc[s, 'SJF-Pred']:.3f} | "
                  f"{benefit(piv.loc[s, 'SJF-Pred'], fcfs, oracle):.0%} | "
                  f"{piv.loc[s, 'SRTF']:.3f} | {piv.loc[s, 'MLFQ-Pred']:.3f} |")


def ageing() -> None:
    head("E4 ageing threshold (swap)")
    e4 = load("e4_starvation_valve")
    fcfs = e4[e4["policy"] == "FCFS"].iloc[0]
    oracle = e4[e4["policy"] == "SJF-Oracle"].iloc[0]
    for _, r in e4.iterrows():
        thr = "ref" if pd.isna(r["starvation_s"]) else ("off" if r["starvation_s"] > 1e8
                                                        else f"{r['starvation_s']:g} s")
        b = benefit(r["norm_latency_mean"], fcfs["norm_latency_mean"], oracle["norm_latency_mean"])
        print(f"{r['policy']:12s} {thr:>6s}  NL {r['norm_latency_mean']:.3f} ({b:4.0%} of oracle)  "
              f"max wait {r['max_wait']:.1f} s ({r['max_wait'] / fcfs['max_wait']:.2f}x FCFS)")


def chunking() -> None:
    head("E5 chunked prefill (bimodal, recompute)")
    e5 = load("e5_chunked_prefill")
    for p in POLICIES:
        d = e5[e5["policy"] == p].set_index("chunk_label")
        off, c64 = d.loc["off"], d.loc["64"]
        best = d["throughput_tok_s"].idxmax()
        print(f"{p:12s} TPOT off {off['tpot_mean']:.4f} -> 64: {c64['tpot_mean']:.4f} "
              f"(TPOT {c64['tpot_mean']/off['tpot_mean'] - 1:+.0%})  "
              f"throughput off {off['throughput_tok_s']:.0f}, best {d['throughput_tok_s'].max():.0f} "
              f"at chunk {best}, 64: {c64['throughput_tok_s']:.0f}")


def preemption_cost() -> None:
    head("E6 preemption cost (default workload, 10 req/s)")
    e6 = load("e6_preemption_cost")
    piv = e6.pivot(index="policy", columns="cost_label", values="norm_latency_mean")
    tput = e6.pivot(index="policy", columns="cost_label", values="throughput_tok_s")
    cols = ["swap x0.1", "swap x1", "swap x10", "recompute", "swap x100"]
    print("NL:        " + "  ".join(f"{c:>10s}" for c in cols))
    for p in piv.index:
        print(f"{p:12s}" + "  ".join(f"{piv.loc[p, c]:10.3f}" for c in cols))
    print("tok/s:")
    for p in tput.index:
        print(f"{p:12s}" + "  ".join(f"{tput.loc[p, c]:10.0f}" for c in cols))


def workload_mix() -> None:
    head("E2 workload mix and goodput by class (recompute)")
    e2 = load("e2_workload_mix")
    d = e2[e2["preemption_mode"] == "recompute"]
    for wl in ("default", "bimodal"):
        x = d[d["workload"] == wl].set_index("policy")
        print(f"{wl}: SJF-Oracle NL {x.loc['FCFS', 'norm_latency_mean']/x.loc['SJF-Oracle', 'norm_latency_mean']:.1f}x better than FCFS")
    x = d[d["workload"] == "default"].set_index("policy")
    print("| Policy | Interactive goodput | Batch goodput | Overall |")
    print("|---|---|---|---|")
    for p in POLICIES:
        r = x.loc[p]
        print(f"| {p} | {100*r['goodput_interactive']:.1f}% | {100*r['goodput_batch']:.1f}% | "
              f"{100*r['goodput_frac']:.1f}% |")


def validation() -> None:
    path = RESULTS / "validation_sim_vs_proxy.csv"
    if not path.exists():
        return
    head("Simulator vs proxy consistency check (mock backend)")
    v = pd.read_csv(path)
    print(v[["policy", "repeats", "proxy_norm_latency_mean", "proxy_norm_latency_min",
             "proxy_norm_latency_max", "sim_norm_latency_mean", "proxy_vs_fcfs",
             "sim_vs_fcfs"]].round(4).to_string(index=False))
    print(f"largest FCFS-ratio gap: {(v['proxy_vs_fcfs'] - v['sim_vs_fcfs']).abs().max():.3f}")
    print(f"rank agreement: {list(v.sort_values('proxy_norm_latency_mean')['policy'])} vs "
          f"{list(v.sort_values('sim_norm_latency_mean')['policy'])}")
    fcfs = v.set_index("policy").loc["FCFS"]
    print(f"absolute FCFS NL: proxy {fcfs['proxy_norm_latency_mean']:.3f} vs "
          f"sim {fcfs['sim_norm_latency_mean']:.3f}")


if __name__ == "__main__":
    for section in (capacity, headline, robustness, prediction, ageing, chunking,
                    preemption_cost, workload_mix, validation):
        section()
