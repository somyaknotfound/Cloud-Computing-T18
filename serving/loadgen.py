"""Load generator and simulator-vs-proxy consistency check.

Owned by: Somyak Mohanta (241CS257)

    python serving/loadgen.py                 # the four non-preemptive policies, mock backend
    python serving/loadgen.py --policies fcfs sjf-pred --repeats 1
    python serving/loadgen.py --backend ollama --model qwen2.5:0.5b

For each policy this starts `serving/proxy.py` in a subprocess, replays the *same seeded
trace* against it over HTTP with Poisson arrivals, collects the server's per-request
timings, then runs the simulator on that identical trace and compares.

**What this does and does not prove.** With the default `mock` backend there is no model
and no GPU: the backend sleeps for as long as the cost model says the work would take. It
has no KV cache and its prefills do not block anyone else's decoding, so absolute latencies
will not match the simulator's, and the comparison deliberately does not claim they should.
This is a *consistency check*, not a hardware validation. What it checks is that the policy
code ranks requests the same way once it is lifted out of the simulator into a real
asynchronous server with real clocks and real concurrency: the same policies should come
out in the same order, with similar ratios to FCFS.

Only non-preemptive policies are checked, because the proxy does not implement preemption.
Wall-clock runs are not bit-reproducible, so each policy is replayed `--repeats` times and
the spread is reported alongside the mean.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.engine import EngineConfig, simulate           # noqa: E402
from src.metrics import aggregate, to_frame             # noqa: E402
from src.policies import ALL_POLICIES, build            # noqa: E402
from src.workload import generate, load_workload_config # noqa: E402

RESULTS = ROOT / "results"
PORT = 8123
MAX_BATCH = 8
N_REQUESTS = 120
RATE = 6.0
MAX_OUTPUT = 300     # keeps a full validation run to a few minutes
PRED_SIGMA = 1.0     # a realistic predictor, so SJF-Pred and SJF-Oracle actually differ
REPEATS = 3
VALIDATED = [p for p in ALL_POLICIES if not build(p).preemptive]


def make_trace(n=N_REQUESTS, rate=RATE, seed=42):
    """One trace, reused for every policy, every repeat, and the simulator."""
    reqs = generate("default", num_requests=n, arrival_rate=rate, seed=seed,
                    predictor_error=PRED_SIGMA)
    for r in reqs:                       # keep wall-clock runtime sane
        r.output_len = min(r.output_len, MAX_OUTPUT)
        r.prompt_len = min(r.prompt_len, 1024)
    return reqs


async def _wait_healthy(url: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as c:
        while time.time() < deadline:
            try:
                r = await c.get(f"{url}/health", timeout=2.0)
                if r.status_code == 200:
                    return r.json()
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError(f"proxy at {url} did not come up within {timeout}s")


async def _fire(client, url, req, t0):
    """Send one request at its scheduled arrival time and drain the token stream."""
    delay = req.arrival - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"rid": req.rid, "prompt_tokens": req.prompt_len,
               "max_tokens": req.output_len, "sla_class": req.sla_class,
               "predicted_len": req.predicted_len}
    async with client.stream("POST", f"{url}/generate", json=payload, timeout=600.0) as r:
        async for _ in r.aiter_bytes():
            pass


async def run_against_proxy(url: str, reqs) -> pd.DataFrame:
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await client.post(f"{url}/reset")
        await asyncio.gather(*[_fire(client, url, r, t0) for r in reqs])
        data = (await client.get(f"{url}/metrics", timeout=30.0)).json()
    return pd.DataFrame(data["records"])


def proxy_metrics(df: pd.DataFrame, slo: dict) -> dict:
    """Same metric definitions as src/metrics.py, applied to the server's own timings."""
    queue_delay = df["start"] - df["arrival"]
    d = pd.DataFrame({
        "rid": df["rid"],
        "sla_class": df["sla_class"],
        "arrival": df["arrival"],
        "prompt_len": df["prompt_len"],
        "output_len": df["output_len"],
        "predicted_len": df["output_len"],
        "queue_delay": queue_delay,
        "wait": queue_delay,             # no preemption here, so one stint in the queue
        "ttft": df["first_token"] - df["arrival"],
        "tpot": np.where(df["output_len"] > 1,
                         (df["finish"] - df["first_token"]) / (df["output_len"] - 1),
                         np.nan),
        "e2e": df["finish"] - df["arrival"],
        "norm_latency": (df["finish"] - df["arrival"]) / df["output_len"],
        "preemptions": 0,
        "finish_time": df["finish"],
    })
    return aggregate(d, slo)


def run_simulator(policy_name: str, slo: dict) -> dict:
    """Simulator configured to match the proxy: same batch cap, KV never the bottleneck."""
    cfg = EngineConfig.load(max_batch_size=MAX_BATCH, kv_blocks_total=1_000_000)
    done, _ = simulate(make_trace(), build(policy_name), config=cfg, slo=slo)
    return aggregate(to_frame(done), slo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policies", nargs="*", default=VALIDATED, choices=VALIDATED)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--backend", default="mock", choices=["mock", "ollama"])
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    slo = load_workload_config()["slo"]
    url = f"http://127.0.0.1:{args.port}"
    rows = []

    for policy_name in args.policies:
        cmd = [sys.executable, "-m", "uvicorn", "serving.proxy:app",
               "--port", str(args.port), "--log-level", "warning"]
        env = {**os.environ, "POLICY": policy_name, "BACKEND": args.backend,
               "MAX_BATCH": str(MAX_BATCH), "OLLAMA_MODEL": args.model}
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
        try:
            health = asyncio.run(_wait_healthy(url))
            runs = []
            t0 = time.perf_counter()
            for _ in range(args.repeats):
                df = asyncio.run(run_against_proxy(url, make_trace()))
                runs.append(proxy_metrics(df, slo))
            wall = time.perf_counter() - t0
            sim = run_simulator(policy_name, slo)

            nl = np.array([r["norm_latency_mean"] for r in runs])
            rows.append({
                "policy": health["policy"],
                "backend": args.backend,
                "n": N_REQUESTS,
                "repeats": args.repeats,
                "proxy_norm_latency_mean": nl.mean(),
                "proxy_norm_latency_min": nl.min(),
                "proxy_norm_latency_max": nl.max(),
                "proxy_ttft_p99": np.mean([r["ttft_p99"] for r in runs]),
                "proxy_max_wait": np.mean([r["max_wait"] for r in runs]),
                "proxy_throughput_tok_s": np.mean([r["throughput_tok_s"] for r in runs]),
                "sim_norm_latency_mean": sim["norm_latency_mean"],
                "sim_ttft_p99": sim["ttft_p99"],
                "sim_max_wait": sim["max_wait"],
                "sim_throughput_tok_s": sim["throughput_tok_s"],
                "wall_s": wall,
            })
            print(f"{health['policy']:14s} proxy NL={nl.mean():.4f} "
                  f"[{nl.min():.4f}-{nl.max():.4f}]  sim NL={sim['norm_latency_mean']:.4f}  "
                  f"({wall:.0f}s, {args.repeats}x{N_REQUESTS} reqs)")
        finally:
            proc.terminate()
            proc.wait(timeout=15)

    out = pd.DataFrame(rows)
    if len(out) > 1 and (out["policy"] == "FCFS").any():
        # Do the two methods AGREE ON THE RANKING? That is the claim being made.
        rank_proxy = out["proxy_norm_latency_mean"].rank()
        rank_sim = out["sim_norm_latency_mean"].rank()
        rho = float(np.corrcoef(rank_proxy, rank_sim)[0, 1])
        base = out.set_index("policy").loc["FCFS"]
        out["proxy_vs_fcfs"] = out["proxy_norm_latency_mean"] / base["proxy_norm_latency_mean"]
        out["sim_vs_fcfs"] = out["sim_norm_latency_mean"] / base["sim_norm_latency_mean"]
        gap = (out["proxy_vs_fcfs"] - out["sim_vs_fcfs"]).abs().max()
        print(f"\nrank correlation, proxy vs simulator: {rho:.3f}   "
              f"largest FCFS-ratio gap: {gap:.3f}")
        print(out[["policy", "proxy_vs_fcfs", "sim_vs_fcfs"]].round(3).to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "validation_sim_vs_proxy.csv"
    path.write_text(out.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
