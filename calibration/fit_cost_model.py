"""Fit the simulator's cost model to real GPU measurements.

Owned by: Ronak Maharia (241CS149)

    python calibration/fit_cost_model.py calibration/calibration_t4.csv

Reads the CSV produced by `colab_calibrate.ipynb`, takes the median of the repeats at each
setting, and fits:

    prefill:  latency = max(floor, b*n + q*n^2)     n = prompt tokens
    decode:   latency = c + d * batch_size
    swap:     latency = s0 + e * tokens             one way

then writes `cost_model.json`, which is what `src/cost_model.py` loads.

**How, and why not plain least squares and R^2.** The first T4 run broke both (BUILD.md
4.17). Two slow decode measurements were enough to make an ordinary least-squares slope
negative -- bigger batches "faster" -- so the decode and swap lines are Theil-Sen fits (the
median of all pairwise slopes), which shrug off a minority of outliers. And R^2 cannot judge a
law that is *meant* to be nearly flat: decode cost barely depends on batch size, so even a
perfect fit explains little variance. Each law is therefore judged by its **relative error**
against the measured medians; the median error must be at most 10% (MAX_MEDIAN_ERR), and the
worst point is reported alongside it.

Prefill is fitted as a segmented model: for each candidate breakpoint, the floor is the median
of the shorter prompts and (b, q) a non-negative least-squares fit to the longer ones; the
breakpoint with the lowest relative error wins.

Input CSV columns:
    phase        "prefill", "decode" or "swap"
    batch_size   sequences in the batch
    tokens       prefill: prompt tokens;  swap: tokens of KV copied;  decode: unused
    latency_s    measured wall-clock seconds (swap: one direction)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAX_MEDIAN_ERR = 0.10


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust line y = intercept + slope*x: median pairwise slope, then median intercept."""
    if len(x) < 2:
        raise ValueError("need at least two points to fit a line")
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i, j in itertools.combinations(range(len(x)), 2) if x[j] != x[i]]
    slope = float(np.median(slopes))
    return float(np.median(y - slope * x)), slope


def nonneg_lstsq(cols: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Least squares with every coefficient >= 0 (exhaustive over which ones are active)."""
    best, best_err = np.zeros(len(cols)), np.inf
    for k in range(1, len(cols) + 1):
        for subset in itertools.combinations(range(len(cols)), k):
            a = np.column_stack([cols[i] for i in subset])
            coef, *_ = np.linalg.lstsq(a, y, rcond=None)
            if (coef < 0).any():
                continue
            err = float(np.sum((a @ coef - y) ** 2))
            if err < best_err:
                best, best_err = np.zeros(len(cols)), err
                best[list(subset)] = coef
    return best


def rel_errors(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pred - y) / y


def fit_prefill(n: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Segmented fit of max(floor, b*n + q*n^2). Returns (floor, b, q)."""
    order = np.argsort(n)
    n, y = n[order], y[order]
    best, best_err = None, np.inf
    for k in range(1, len(n) - 1):          # at least one floor point, two compute points
        floor = float(np.median(y[:k]))
        b, q = nonneg_lstsq([n[k:], n[k:] ** 2], y[k:])
        err = float(np.sum(rel_errors(np.maximum(floor, b * n + q * n ** 2), y) ** 2))
        if err < best_err:
            best, best_err = (floor, float(b), float(q)), err
    return best


def medians(df: pd.DataFrame, phase: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    m = df[df["phase"] == phase].groupby(key)["latency_s"].median()
    return m.index.to_numpy(float), m.to_numpy(float)


def fit(df: pd.DataFrame) -> dict:
    """All constants plus the quality of each fit."""
    n, y_pre = medians(df, "prefill", "tokens")
    b_size, y_dec = medians(df, "decode", "batch_size")
    k, y_swp = medians(df, "swap", "tokens")
    if not (len(n) and len(b_size) and len(k)):
        raise SystemExit("CSV needs prefill, decode and swap rows (re-run the current notebook)")

    floor, b, q = fit_prefill(n, y_pre)
    c, d = theil_sen(b_size, y_dec)
    s0, e = theil_sen(k, y_swp)

    quality = {}
    for name, x, y, pred in (
            ("prefill", n, y_pre, np.maximum(floor, b * n + q * n ** 2)),
            ("decode", b_size, y_dec, c + d * b_size),
            ("swap", k, y_swp, s0 + e * k)):
        err = rel_errors(pred, y)
        quality[name] = {"median_rel_err": round(float(np.median(err)), 4),
                         "max_rel_err": round(float(err.max()), 4),
                         "worst_at": float(x[err.argmax()]),
                         "points": int(len(x))}
    return {
        "prefill_floor_s": round(floor, 8),
        "prefill_per_token_s": round(b, 12),
        "prefill_per_token2_s": round(q, 15),
        "decode_base_s": round(c, 8),
        "decode_per_seq_s": round(d, 10),
        "swap_base_s": round(s0, 9),
        "swap_per_token_s": round(e, 13),
        "fit_quality": quality,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default=str(HERE / "calibration_t4.csv"))
    ap.add_argument("-o", "--out", default=str(HERE / "cost_model.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"phase", "batch_size", "tokens", "latency_s"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{args.csv} must have columns {sorted(required)}")

    out = fit(df)
    out.update({
        "device": str(df["device"].iloc[0]) if "device" in df else "unknown",
        "model": str(df["model"].iloc[0]) if "model" in df else "unknown",
        "source": f"measured -- fitted from {Path(args.csv).name} by fit_cost_model.py",
    })
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    q = out["fit_quality"]
    print(f"prefill: t = max({out['prefill_floor_s']*1e3:.2f} ms, "
          f"{out['prefill_per_token_s']*1e6:.2f} us*n + {out['prefill_per_token2_s']*1e9:.3f} ns*n^2)")
    print(f"decode:  t = {out['decode_base_s']*1e3:.2f} ms + {out['decode_per_seq_s']*1e6:.1f} us/sequence")
    print(f"swap:    t = {out['swap_base_s']*1e3:.3f} ms + {out['swap_per_token_s']*1e6:.3f} us/token, one way")
    for name, s in q.items():
        print(f"  {name:8s} median error {s['median_rel_err']:6.1%}   worst {s['max_rel_err']:6.1%} "
              f"(at {s['worst_at']:g})   {s['points']} points")
    c, d = out["decode_base_s"], out["decode_per_seq_s"]
    print(f"\ndecoding 32 sequences costs {(c + 32*d)/(c + d):.2f}x one sequence, not 32x")
    print(f"wrote {args.out}")

    bad = [n for n, s in q.items() if s["median_rel_err"] > MAX_MEDIAN_ERR]
    if bad:
        print(f"\n!! WARNING: median error above {MAX_MEDIAN_ERR:.0%} for {bad}. The model does "
              f"not describe this GPU;\n   say so in the report instead of quoting the constants.")


if __name__ == "__main__":
    main()
