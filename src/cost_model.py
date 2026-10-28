"""The GPU cost model: how long one engine iteration takes.

Owned by: Ronak Maharia (241CS149)

This is the only place where the simulator touches physical reality. Every timing in the
whole benchmark comes from the constants in `calibration/cost_model.json`, which
`calibration/fit_cost_model.py` fits to the measurements `colab_calibrate.ipynb` takes on a
real GPU. `source` in that file says whether they are measured or provisional.

Three laws. Their shapes are what the T4 measurements showed, not what we first assumed (a
plain straight line for prefill fitted with a 55% median error -- see BUILD.md 4.17):

  prefill:  t = max(floor, b*n + q*n^2)          n = prompt tokens of one sequence
      Short prompts cost a fixed `floor`: the pass is bound by kernel-launch overhead, not by
      arithmetic. Past ~500 tokens it becomes compute-bound: b*n for the per-token work and
      q*n^2 for attention, where every token attends to every earlier one.

  decode:   t = c + d * batch_size
      One decode pass emits a single token per sequence. Nearly all of it is the fixed cost
      c (streaming the weights, launching kernels), paid whether the batch holds 1 sequence
      or 32; the marginal cost d of one more sequence is small. That is exactly why
      continuous batching wins.

  swap:     t = s0 + e * tokens                   one way, GPU <-> host
      Only used by swap preemption: copying a preempted request's KV cache over PCIe.

Prefill work is passed as *segments* -- (tokens already prefilled, tokens in this pass) for
each sequence -- because attention is quadratic per sequence: eight 256-token prompts cost
far less than one 2048-token prompt, and a chunk of a long prompt attends to everything
before it. A chunk from position s to s+k costs q*((s+k)^2 - s^2), so prefilling a prompt
in chunks adds up to exactly the same compute as prefilling it in one go.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent.parent / "calibration"

Segment = tuple[int, int]   # (tokens of this sequence already prefilled, tokens in this pass)


@dataclass(frozen=True)
class CostModel:
    prefill_floor_s: float       # launch-bound minimum for any prefill pass
    prefill_per_token_s: float   # b -- per-token compute (MLPs, projections)
    prefill_per_token2_s: float  # q -- attention, per (token x earlier token)
    decode_base_s: float         # c -- fixed cost of one decode pass
    decode_per_seq_s: float      # d -- marginal cost of one more sequence in the batch
    swap_base_s: float           # s0 -- fixed cost of one GPU<->host copy
    swap_per_token_s: float      # e -- per token of KV moved, one way
    source: str = "unknown"
    device: str = "unknown"
    model: str = "unknown"

    def prefill_compute(self, segments: list[Segment]) -> float:
        """Compute-bound prefill work for these segments, without the launch floor."""
        total = 0.0
        for start, n in segments:
            if n > 0:
                total += (self.prefill_per_token_s * n
                          + self.prefill_per_token2_s * ((start + n) ** 2 - start ** 2))
        return total

    def prefill_time(self, segments: list[Segment]) -> float:
        """A prefill-only pass over these segments."""
        if not any(n > 0 for _, n in segments):
            return 0.0
        return max(self.prefill_floor_s, self.prefill_compute(segments))

    def decode_time(self, batch_size: int) -> float:
        if batch_size <= 0:
            return 0.0
        return self.decode_base_s + self.decode_per_seq_s * batch_size

    def hybrid_time(self, segments: list[Segment], decode_seqs: int) -> float:
        """A chunked-prefill (Sarathi-Serve) pass: prefill chunks fused with the decodes.

        The fixed cost of a pass is paid once, by the decode step; the chunks only add
        their compute. That saving is the entire point of fusing them.
        """
        if decode_seqs <= 0:
            return self.prefill_time(segments)
        return self.decode_time(decode_seqs) + self.prefill_compute(segments)

    def swap_time(self, tokens: int) -> float:
        """One-way GPU<->host copy of `tokens` tokens of KV cache (swap preemption)."""
        if tokens <= 0:
            return 0.0
        return self.swap_base_s + self.swap_per_token_s * tokens

    def peak_decode_throughput(self, batch_size: int) -> float:
        """Tokens/sec at a given batch size -- used to sanity-check saturation points."""
        return batch_size / self.decode_time(batch_size)

    def scaled_swap(self, factor: float) -> "CostModel":
        """The same model with swapping `factor` times as expensive (experiment E6)."""
        return CostModel(**{**self.__dict__, "swap_base_s": self.swap_base_s * factor,
                            "swap_per_token_s": self.swap_per_token_s * factor})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CostModel":
        path = Path(path) if path else CALIB_DIR / "cost_model.json"
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
