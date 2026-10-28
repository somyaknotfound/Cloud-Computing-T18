"""Workload generation: turns a YAML workload spec into a list of Requests.

Owned by: Pranav Shaji (241CS243)

Every field a scheduler is allowed to look at is separated from the fields only the
simulator knows. `output_len` is ground truth -- a real serving system cannot see it
until the request finishes. Policies that want it must go through `predicted_len`,
which carries a controllable amount of prediction error. That separation is the whole
point of experiment E3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class Request:
    """One inference request moving through the system."""

    rid: int
    arrival: float          # seconds, relative to start of trace
    prompt_len: int         # input tokens -- visible to the scheduler on arrival
    output_len: int         # GROUND TRUTH output tokens; the scheduler must not read this
    sla_class: str          # "interactive" or "batch"
    predicted_len: int      # what a length predictor believes; what SJF-Pred is allowed to use

    # --- mutable state, filled in by the engine as the request is served ---
    generated: int = 0              # tokens emitted so far
    prefilled: int = 0              # tokens of the prefill target processed so far
    prefill_target: int = 0         # tokens this admission must prefill; set by the engine
    blocks: int = 0                 # KV pages currently reserved; engine-owned accounting
    start_time: float | None = None # first moment it entered the running batch
    first_token_time: float | None = None
    finish_time: float | None = None
    preemptions: int = 0
    attained_service: float = 0.0   # GPU-seconds consumed
    wait_time: float = 0.0          # total time in the waiting queue, across preemptions
    queued_at: float = 0.0          # when it last entered the waiting queue
    swapped: bool = False           # KV cache currently parked in host memory

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len

    @property
    def prefill_done(self) -> bool:
        """True once this admission's prefill target is met.

        The target is not always `prompt_len`: a request preempted after generating G
        tokens lost their KV entries too, so on re-admission it must prefill
        `prompt_len + G` tokens before it can resume decoding.
        """
        return self.prefilled >= self.prefill_target

    @property
    def remaining(self) -> int:
        """Ground-truth tokens still to generate. Simulator-only; never read by a policy."""
        return self.output_len - self.generated

    @property
    def predicted_remaining(self) -> int:
        """What a length predictor would say is left. This is what SRTF is allowed to use."""
        return max(1, self.predicted_len - self.generated)

    def kv_tokens(self) -> int:
        """Tokens this request needs KV space for: its prompt plus everything generated."""
        return self.prompt_len + self.generated

    def reset_for_recompute(self) -> None:
        """vLLM-style preemption: drop the KV cache, keep the tokens already shown to the user.

        The request goes back to the waiting queue and must re-prefill everything it had
        (prompt + tokens generated so far, i.e. `kv_tokens()`) before it can resume
        decoding. Tokens already streamed to the user are not lost -- only their KV
        entries are -- so the user sees a stall, not a restart.
        """
        self.prefilled = 0
        self.blocks = 0
        self.preemptions += 1

    def swap_out(self) -> None:
        """FastServe-style preemption: park the KV cache in host memory.

        Unlike recompute, nothing has to be redone on return -- prefill progress is kept and
        only the GPU pages are released. The price is the transfer itself, which the engine
        charges in both directions.
        """
        self.blocks = 0
        self.swapped = True
        self.preemptions += 1


def _lognormal(rng: np.random.Generator, spec: dict, n: int) -> np.ndarray:
    """Draw n lognormal lengths, clipped to [min, max], as integers."""
    raw = rng.lognormal(mean=spec["mean_log"], sigma=spec["sigma_log"], size=n)
    return np.clip(np.rint(raw), spec["min"], spec["max"]).astype(int)


def load_workload_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else CONFIG_DIR / "workloads.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def generate(
    name: str = "default",
    *,
    arrival_rate: float | None = None,
    num_requests: int | None = None,
    seed: int | None = None,
    predictor_error: float = 0.0,
    config: dict | None = None,
) -> list[Request]:
    """Build a reproducible request trace.

    predictor_error is the sigma of multiplicative lognormal noise on the length
    prediction. 0.0 means a perfect oracle; 0.5 means predictions are typically
    within about +/-65% of the truth. Sweeping it answers "how good does a length
    predictor have to be before SJF is worth it?".
    """
    cfg = config or load_workload_config()
    if name not in cfg:
        raise KeyError(f"unknown workload {name!r}; have {sorted(k for k in cfg if k != 'slo')}")
    spec = cfg[name]

    rate = arrival_rate if arrival_rate is not None else spec["arrival_rate"]
    n = num_requests if num_requests is not None else spec["num_requests"]
    rng = np.random.default_rng(seed if seed is not None else spec["seed"])

    # Poisson arrivals == exponential gaps.
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=n))

    if spec.get("bimodal"):
        is_short = rng.random(n) < spec["short_fraction"]
        prompts = np.where(is_short,
                           _lognormal(rng, spec["prompt_len"], n),
                           _lognormal(rng, spec["long_prompt_len"], n))
        outputs = np.where(is_short,
                           _lognormal(rng, spec["output_len"], n),
                           _lognormal(rng, spec["long_output_len"], n))
    else:
        prompts = _lognormal(rng, spec["prompt_len"], n)
        outputs = _lognormal(rng, spec["output_len"], n)

    p_interactive = spec["class_mix"]["interactive"]
    classes = np.where(rng.random(n) < p_interactive, "interactive", "batch")

    if predictor_error <= 0:
        predictions = outputs.copy()
    else:
        noise = rng.lognormal(mean=0.0, sigma=predictor_error, size=n)
        predictions = np.maximum(1, np.rint(outputs * noise)).astype(int)

    return [
        Request(
            rid=i,
            arrival=float(arrivals[i]),
            prompt_len=int(prompts[i]),
            output_len=int(outputs[i]),
            sla_class=str(classes[i]),
            predicted_len=int(predictions[i]),
        )
        for i in range(n)
    ]


def summarize(requests: list[Request]) -> dict:
    """Descriptive stats for the trace, for the report's workload table."""
    prompts = np.array([r.prompt_len for r in requests])
    outputs = np.array([r.output_len for r in requests])
    span = requests[-1].arrival - requests[0].arrival
    return {
        "n": len(requests),
        "arrival_rate_actual": len(requests) / span if span > 0 else float("nan"),
        "prompt_median": float(np.median(prompts)),
        "prompt_p99": float(np.percentile(prompts, 99)),
        "output_median": float(np.median(outputs)),
        "output_p99": float(np.percentile(outputs, 99)),
        "output_max": int(outputs.max()),
        "total_output_tokens": int(outputs.sum()),
        "interactive_frac": sum(r.sla_class == "interactive" for r in requests) / len(requests),
    }
