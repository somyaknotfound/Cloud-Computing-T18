"""Experiment driver: runs the sweeps and writes results/*.csv.

Owned by: Somyak Mohanta (241CS257)

    python -m src.runner --all --jobs 8     # everything, in parallel
    python -m src.runner e1 e3              # just those experiments
    python -m src.runner --all --quick      # smaller traces, for a fast check
    python -m src.runner --all --verify     # re-run and compare byte-for-byte with results/

Every run rebuilds its workload from a fixed seed, so results are reproducible and two
policies are always compared on the *identical* request trace. That matters: with a
heavy-tailed length distribution, two different random traces differ more than two
scheduling policies do. `--jobs` only changes how fast the runs happen: results are
collected in task order, so the CSVs are identical to a serial run, and `--verify` proves
it.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .cost_model import CostModel
from .engine import EngineConfig, simulate
from .metrics import aggregate, to_frame
from .policies import ALL_POLICIES, build
from .workload import generate, load_workload_config

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The modelled engine saturates near 5.7 req/s (see e0_capacity.csv). Rates below that
# leave the queue empty and no policy can differentiate itself; rates above it are where
# scheduling actually decides who waits. At 10 req/s, 600 requests arrive over ~60 s:
# every comparison is a one-minute overload burst followed by the drain.
LOADED_RATE = 10.0
N_REQUESTS = 600
QUICK_REQUESTS = 200

# The cost of preempting a request decides whether preemptive policies work at all, so
# most experiments run under both. See src/engine.py::_evict.
MODES = ("recompute", "swap")

# E7 repeats the headline comparison on fresh traces. The single-trace sweeps use the
# workload's own seed (42), which is deliberately not in this list.
SEEDS = tuple(range(1, 11))
T95 = 2.262  # two-sided 95% Student-t critical value for len(SEEDS) - 1 = 9 dof


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed directly so the project does not need SciPy.

    This is the honest way to describe a length predictor's quality. Sigma is an internal
    knob of our noise model; rank correlation is what you could actually measure for a
    real predictor, and SJF only ever cares about the *order* it induces, not the values.
    """
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def run_point(policy_name: str, *, workload="default", rate=LOADED_RATE, n=N_REQUESTS,
              seed=None, predictor_error=0.0, mode="recompute", chunk=None,
              swap_scale=1.0, policy_kwargs=None, label=None) -> dict:
    """One (policy, trace, engine-config) point. Returns a row of aggregated metrics."""
    wl_cfg = load_workload_config()
    slo = wl_cfg["slo"]
    seed = seed if seed is not None else wl_cfg[workload]["seed"]

    cm = CostModel.load()
    cm = cm.scaled_swap(swap_scale)
    cfg = EngineConfig.load(preemption_mode=mode, chunked_prefill=chunk is not None,
                            prefill_chunk_size=chunk)

    reqs = generate(workload, num_requests=n, arrival_rate=rate, seed=seed,
                    predictor_error=predictor_error)
    rank_corr = _spearman(np.array([r.predicted_len for r in reqs]),
                          np.array([r.output_len for r in reqs]))
    policy = build(policy_name, **(policy_kwargs or {}))
    done, stats = simulate(reqs, policy, cost_model=cm, config=cfg, slo=slo)

    row = aggregate(to_frame(done), slo, label={
        "policy": policy.name,
        "workload": workload,
        "arrival_rate": rate,
        "seed": seed,
        "preemption_mode": mode,
        "swap_scale": swap_scale,
        "predictor_error": predictor_error,
        "predictor_rank_corr": rank_corr,
        "chunked_prefill": cfg.chunked_prefill,
        "prefill_chunk_size": cfg.prefill_chunk_size if cfg.chunked_prefill else 0,
        **(label or {}),
    })
    row["preemptions"] = stats.preemptions
    row["recomputed_tokens"] = stats.recomputed_tokens
    row["swapped_tokens"] = stats.swapped_tokens
    row["iterations"] = stats.iterations
    row["prefill_iter_frac"] = stats.prefill_iterations / max(1, stats.iterations)
    return row


def _task(kwargs: dict) -> dict:
    return run_point(**kwargs)


def run_tasks(tasks: list[dict], jobs: int) -> list[dict]:
    """Run every task; the result order is the task order, whatever `jobs` is."""
    if jobs <= 1:
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_task, tasks, chunksize=2))


# ---------- experiments: each returns the list of points to run ----------

def e0_capacity(n: int) -> list[dict]:
    """Where does this engine saturate? Establishes the x-axis for everything else."""
    return [dict(policy_name="fcfs", rate=rate, n=n)
            for rate in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20)]


def e1_load_sweep(n: int) -> list[dict]:
    """E1: latency vs offered load, per policy, under both preemption mechanisms."""
    return [dict(policy_name=p, rate=rate, n=n, mode=mode)
            for mode in MODES
            for rate in (2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
            for p in ALL_POLICIES]


def e2_workload_mix(n: int) -> list[dict]:
    """E2: does the winner change when short chats share the GPU with long batch jobs?"""
    return [dict(policy_name=p, workload=wl, n=n, mode=mode)
            for mode in MODES
            for wl in ("default", "bimodal")
            for p in ALL_POLICIES]


def e3_prediction_error(n: int) -> list[dict]:
    """E3: how accurate must a length predictor be before using one is worth it?

    FCFS, SJF-Oracle and MLFQ-LAS are included as flat reference lines -- none of them
    reads the prediction, so none of them may move with sigma. MLFQ-LAS staying flat is
    the evidence that it really is predictor-free.

    Sigma runs all the way to 8.0 deliberately. Milder noise never actually costs SJF
    its lead, so a sweep that stopped at sigma=2 would report "prediction always helps"
    and miss where the crossover really is.
    """
    sigmas = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    pols = ("sjf-pred", "srtf", "mlfq-pred", "fcfs", "sjf-oracle", "mlfq")
    return [dict(policy_name=p, predictor_error=s, n=n, mode=mode)
            for mode in MODES for s in sigmas for p in pols]


def e4_starvation_valve(n: int) -> list[dict]:
    """E4: MLFQ's ageing threshold, trading mean latency against the worst case.

    Run under swap preemption, the only mechanism under which MLFQ is viable (E6). Set
    the threshold too low and every request is promoted to level 0, which is just FCFS
    with extra steps. Set it too high and long requests starve like they do under SJF.
    """
    tasks = [dict(policy_name=p, n=n, mode="swap",
                  policy_kwargs={"starvation_s": s}, label={"starvation_s": s})
             for p in ("mlfq", "mlfq-pred")
             for s in (2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 1e9)]
    tasks += [dict(policy_name=p, n=n, mode="swap", label={"starvation_s": float("nan")})
              for p in ("fcfs", "sjf-oracle")]
    return tasks


def e5_chunked_prefill(n: int) -> list[dict]:
    """E5: Sarathi-Serve's chunked prefill -- an engine mechanism, not a policy.

    Chunking splits a long prompt across several iterations and fuses each piece with the
    ongoing decodes, so decoding never stalls for the whole length of a long prefill.

    Run on the *bimodal* workload, because that is the only one where it can matter: the
    chunk size has to be smaller than the prompts to do anything at all, and the default
    workload's median prompt (244 tokens) is under even the smallest chunk we sweep. The
    sweep is over chunk size rather than a simple on/off, because the size is precisely
    the TTFT-versus-TPOT dial.
    """
    tasks = [dict(policy_name=p, workload="bimodal", n=n, label={"chunk_label": "off"})
             for p in ALL_POLICIES]
    tasks += [dict(policy_name=p, workload="bimodal", n=n, chunk=chunk,
                   label={"chunk_label": str(chunk)})
              for chunk in (1024, 512, 256, 128, 64) for p in ALL_POLICIES]
    return tasks


def e6_preemption_cost(n: int) -> list[dict]:
    """E6: how cheap must preemption be before preemptive policies pay off?

    The swap cost is scaled from a tenth of its measured value up to 100x, which is
    past the point where it costs as much as recompute does. This also answers "what if
    the swap constant is wrong?" -- and the answer is that which preemptive policy wins
    depends on it, which is why the calibration notebook measures it.
    """
    pols = ("fcfs", "sjf-oracle", "srtf", "mlfq", "mlfq-pred")
    tasks = [dict(policy_name=p, n=n, mode="swap", swap_scale=k,
                  label={"cost_label": f"swap x{k:g}"})
             for k in (0.1, 1.0, 10.0, 100.0) for p in pols]
    tasks += [dict(policy_name=p, n=n, mode="recompute", label={"cost_label": "recompute"})
              for p in pols]
    return tasks


def e7_robustness(n: int) -> list[dict]:
    """E7: is the headline comparison a property of the policies, or of trace #42?

    Ten fresh seeds, at half, one and two times the usual trace length (the burst length
    is a free parameter of an overload experiment, so the ranking must not depend on it).
    """
    return [dict(policy_name=p, n=size, seed=seed, mode=mode)
            for mode in MODES
            for size in (n // 2, n, 2 * n)
            for seed in SEEDS
            for p in ALL_POLICIES]


SUMMARY_METRICS = {           # metric -> which direction is better
    "norm_latency_mean": "lower",
    "ttft_p50": "lower",
    "max_wait": "lower",
    "goodput_frac": "higher",
    "throughput_tok_s": "higher",
}


def summarize_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over seeds, and the paired ratio to FCFS on the same trace."""
    rows = []
    for (mode, size, policy), g in df.groupby(["preemption_mode", "n_requests", "policy"],
                                              sort=True):
        base = df[(df["preemption_mode"] == mode) & (df["n_requests"] == size)
                  & (df["policy"] == "FCFS")].set_index("seed")
        g = g.set_index("seed").loc[base.index]
        for metric, better in SUMMARY_METRICS.items():
            v = g[metric].to_numpy()
            ratio = v / base[metric].to_numpy()
            wins = v < base[metric].to_numpy() if better == "lower" else v > base[metric].to_numpy()
            rows.append({
                "preemption_mode": mode, "n_requests": size, "policy": policy,
                "metric": metric, "seeds": len(v),
                "mean": v.mean(), "ci95": T95 * v.std(ddof=1) / np.sqrt(len(v)),
                "ratio_vs_fcfs": ratio.mean(),
                "ratio_ci95": T95 * ratio.std(ddof=1) / np.sqrt(len(v)),
                "seeds_better_than_fcfs": int(wins.sum()),
            })
    return pd.DataFrame(rows)


EXPERIMENTS = {
    "e0": ("e0_capacity", e0_capacity),
    "e1": ("e1_load_sweep", e1_load_sweep),
    "e2": ("e2_workload_mix", e2_workload_mix),
    "e3": ("e3_prediction_error", e3_prediction_error),
    "e4": ("e4_starvation_valve", e4_starvation_valve),
    "e5": ("e5_chunked_prefill", e5_chunked_prefill),
    "e6": ("e6_preemption_cost", e6_preemption_cost),
    "e7": ("e7_robustness", e7_robustness),
}


def _csv(df: pd.DataFrame) -> str:
    # "\n" explicitly: the default is os.linesep, which would make the files differ
    # between Windows and Linux.
    return df.to_csv(index=False, lineterminator="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LLM scheduling benchmark sweeps.")
    # default=None rather than []: with nargs="*" argparse validates the default against
    # `choices`, and an empty list is not a valid choice.
    ap.add_argument("experiments", nargs="*", choices=list(EXPERIMENTS), default=None)
    ap.add_argument("--all", action="store_true", help="run every experiment")
    ap.add_argument("--quick", action="store_true", help="smaller traces, for a fast check")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (output unchanged)")
    ap.add_argument("--verify", action="store_true",
                    help="re-run and compare with results/*.csv instead of writing")
    args = ap.parse_args()

    chosen = list(EXPERIMENTS) if args.all or not args.experiments else args.experiments
    n = QUICK_REQUESTS if args.quick else N_REQUESTS
    RESULTS.mkdir(exist_ok=True)

    cm = CostModel.load()
    if cm.source.startswith("PROVISIONAL"):
        print("!! cost model is PROVISIONAL (not measured). Run calibration/colab_calibrate.ipynb\n"
              "   on Colab, then calibration/fit_cost_model.py, before quoting these numbers.\n")

    mismatches = 0
    for key in chosen:
        name, fn = EXPERIMENTS[key]
        t0 = time.perf_counter()
        tasks = fn(n)
        df = pd.DataFrame(run_tasks(tasks, args.jobs))
        outputs = {name: df}
        if key == "e7":
            outputs["e7_summary"] = summarize_seeds(df)

        for out_name, out_df in outputs.items():
            path = RESULTS / f"{out_name}.csv"
            text = _csv(out_df)
            if args.verify:
                same = path.exists() and path.read_bytes().replace(b"\r\n", b"\n") == text.encode()
                mismatches += not same
                status = "identical" if same else "DIFFERS"
            else:
                path.write_text(text, encoding="utf-8", newline="")
                status = "written"
            print(f"{key}  {out_name:24s} {len(out_df):4d} rows  "
                  f"{time.perf_counter()-t0:6.1f}s  {status}")

    if args.verify:
        print(f"\n{'all outputs identical' if not mismatches else f'{mismatches} output(s) differ'}")
        sys.exit(1 if mismatches else 0)
    print(f"\nwrote {len(chosen)} experiment(s) to {RESULTS}")
    print("next: python -m src.plots")


if __name__ == "__main__":
    main()
