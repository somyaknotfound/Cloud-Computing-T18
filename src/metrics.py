"""Metrics: turns a finished request trace into the numbers the report quotes.

Owned by: Pranav Shaji (241CS243)

Definitions follow the LLM-serving literature (Orca, vLLM, FastServe, Sarathi-Serve)
rather than classic OS scheduling, because a request here is a *stream* of tokens,
not a single job that finishes all at once:

  TTFT   time to first token   = first_token_time - arrival        (responsiveness)
  TPOT   time per output token = (finish - first_token) / (out-1)  (streaming smoothness)
  E2E    end-to-end latency    = finish - arrival
  NL     normalized latency    = E2E / output_len

NL is the headline metric in this literature: comparing raw E2E across requests is
unfair because a 1000-token answer *should* take longer than a 10-token one. NL
divides that out, so it isolates what the scheduler did rather than how big the job was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .workload import Request

PERCENTILES = (50, 90, 99)


def to_frame(requests: list[Request]) -> pd.DataFrame:
    """Per-request table. Only completed requests are included."""
    rows = []
    for r in requests:
        if r.finish_time is None or r.first_token_time is None:
            continue
        e2e = r.finish_time - r.arrival
        decode_span = r.finish_time - r.first_token_time
        rows.append({
            "rid": r.rid,
            "sla_class": r.sla_class,
            "arrival": r.arrival,
            "prompt_len": r.prompt_len,
            "output_len": r.output_len,
            "predicted_len": r.predicted_len,
            "queue_delay": r.start_time - r.arrival,
            # Total time spent waiting, including every stint back in the queue after a
            # preemption. A preemptive policy can admit a request at once and starve it
            # later; queue_delay (first admission only) would never see that.
            "wait": r.wait_time,
            "ttft": r.first_token_time - r.arrival,
            # TPOT is the average gap *between* tokens, so a single-token answer has no
            # TPOT at all. Recording 0.0 there would silently flatter every policy's
            # percentiles, so it is NaN and excluded from TPOT statistics instead.
            "tpot": (decode_span / (r.output_len - 1)) if r.output_len > 1 else float("nan"),
            "e2e": e2e,
            "norm_latency": e2e / r.output_len,
            "preemptions": r.preemptions,
            "finish_time": r.finish_time,
        })
    if not rows:
        raise ValueError("no completed requests -- did the engine run?")
    return pd.DataFrame(rows)


def jains_index(x: np.ndarray) -> float:
    """Jain's fairness index over per-request normalized latency.

    1.0 means every request suffered exactly the same slowdown; lower means the
    scheduler treated some requests much worse than others. This is where SJF-style
    policies pay for their low average latency.
    """
    x = np.asarray(x, dtype=float)
    denom = len(x) * np.sum(x ** 2)
    return float(np.sum(x) ** 2 / denom) if denom > 0 else float("nan")


def _slo_ok(df: pd.DataFrame, slo: dict) -> pd.Series:
    """A request is 'good' only if it meets BOTH the TTFT and TPOT target for its class."""
    ttft_target = df["sla_class"].map(lambda c: slo[c]["ttft_s"])
    tpot_target = df["sla_class"].map(lambda c: slo[c]["tpot_s"])
    # Single-token answers have no TPOT to violate, so they pass that half by default.
    tpot_ok = (df["tpot"] <= tpot_target) | df["tpot"].isna()
    return (df["ttft"] <= ttft_target) & tpot_ok


def aggregate(df: pd.DataFrame, slo: dict, *, label: dict | None = None) -> dict:
    """Collapse a per-request table into one row of summary statistics."""
    makespan = df["finish_time"].max() - df["arrival"].min()
    out = dict(label or {})
    out["n_requests"] = len(df)

    for col in ("ttft", "tpot", "e2e", "norm_latency", "queue_delay", "wait"):
        vals = df[col].dropna().to_numpy()          # drops single-token TPOT rows
        if len(vals) == 0:                          # every request was single-token
            out[f"{col}_mean"] = float("nan")
            for p in PERCENTILES:
                out[f"{col}_p{p}"] = float("nan")
            continue
        out[f"{col}_mean"] = float(vals.mean())
        for p in PERCENTILES:
            out[f"{col}_p{p}"] = float(np.percentile(vals, p))

    out["throughput_req_s"] = len(df) / makespan
    out["throughput_tok_s"] = float(df["output_len"].sum()) / makespan
    out["makespan_s"] = float(makespan)

    good = _slo_ok(df, slo)
    out["goodput_frac"] = float(good.mean())
    for cls in sorted(df["sla_class"].unique()):
        sel = df["sla_class"] == cls
        out[f"goodput_{cls}"] = float(good[sel].mean())
        out[f"norm_latency_mean_{cls}"] = float(df.loc[sel, "norm_latency"].mean())

    out["jain_fairness"] = jains_index(df["norm_latency"].to_numpy())
    out["max_queue_delay"] = float(df["queue_delay"].max())   # first admission only
    out["max_wait"] = float(df["wait"].max())                 # starvation witness
    out["preemptions_total"] = int(df["preemptions"].sum())
    return out


def compare_table(rows: list[dict], sort_by: str = "norm_latency_mean") -> pd.DataFrame:
    """Tidy summary frame across policies, for the report."""
    df = pd.DataFrame(rows)
    return df.sort_values(sort_by).reset_index(drop=True) if sort_by in df else df
