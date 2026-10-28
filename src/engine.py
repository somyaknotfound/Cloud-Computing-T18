"""Discrete-event simulator of an LLM inference engine with continuous batching.

Owned by: Ronak Maharia (241CS149)

This models one GPU worker running an Orca/vLLM-style engine. The thing to understand
before reading the code is that **an LLM server does not run one request at a time.**

A request has two phases:

  PREFILL   process the whole prompt in one parallel pass, and emit token #1.
            Cost grows with the number of prompt tokens.
  DECODE    emit one more token. Cost is dominated by streaming the model weights
            from HBM once, which is paid whether the batch holds 1 sequence or 32.

Because decode is memory-bound, running 32 sequences in one pass costs barely more than
running 1. So the engine keeps a *running batch* and, between every single-token
iteration, admits newly arrived requests and retires finished ones. That is continuous
(iteration-level) batching, and it is why LLM scheduling is not classic OS scheduling:
the scheduler is not choosing who runs next, it is choosing **who is in the batch**.

The hard constraint is not CPU time, it is the **KV cache**. Every sequence in the batch
holds one entry per token it has seen, in fixed-size pages (vLLM's PagedAttention). When
those pages run out, the engine must evict somebody -- and *which* request gets evicted
is where the scheduling policy shows up a second time.

Every timing comes from `cost_model.py`, whose constants are fitted to GPU measurements
(see `calibration/`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .cost_model import CostModel
from .policies import Policy, SchedulerState
from .workload import Request

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class EngineConfig:
    max_batch_size: int = 32
    block_size: int = 16
    kv_blocks_total: int = 1024
    max_prefill_tokens: int = 2048
    chunked_prefill: bool = False
    prefill_chunk_size: int = 512
    preemption_mode: str = "recompute"
    max_iterations: int = 2_000_000

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "EngineConfig":
        path = Path(path) if path else CONFIG_DIR / "engine.yaml"
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in known}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


@dataclass
class IterationRecord:
    """One row of the engine trace, for occupancy plots and debugging."""

    t: float
    kind: str  # "prefill" | "decode" | "hybrid"
    batch_size: int
    prefill_tokens: int
    kv_blocks_used: int
    n_waiting: int


@dataclass
class RunStats:
    iterations: int = 0
    prefill_iterations: int = 0
    decode_iterations: int = 0
    hybrid_iterations: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    swapped_tokens: int = 0
    busy_time: float = 0.0
    idle_time: float = 0.0
    trace: list[IterationRecord] = field(default_factory=list)


class Engine:
    """Simulates the engine end to end and returns the completed requests."""

    def __init__(
        self,
        policy: Policy,
        cost_model: CostModel,
        config: EngineConfig,
        slo: dict,
        *,
        record_trace: bool = False,
    ):
        self.policy = policy
        self.cm = cost_model
        self.cfg = config
        self.slo = slo
        self.record_trace = record_trace

        if config.preemption_mode not in ("recompute", "swap"):
            raise ValueError(f"preemption_mode must be 'recompute' or 'swap', "
                             f"got {config.preemption_mode!r}")

        self.now = 0.0
        self.overhead_s = 0.0  # pending swap-transfer time, charged to the next pass
        self.kv_used = 0  # blocks
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.stats = RunStats()

    # ---------- KV cache accounting ----------

    def _blocks(self, tokens: int) -> int:
        return math.ceil(tokens / self.cfg.block_size)

    @property
    def kv_free(self) -> int:
        return self.cfg.kv_blocks_total - self.kv_used

    def _blocks_needed(self, r: Request) -> int:
        """Pages required to hold this request's prompt plus everything it has generated."""
        return self._blocks(r.kv_tokens())

    def _state(self) -> SchedulerState:
        return SchedulerState(
            now=self.now,
            kv_blocks_free=self.kv_free,
            kv_blocks_total=self.cfg.kv_blocks_total,
            running_count=len(self.running),
            slo=self.slo,
        )

    # ---------- preemption ----------

    def _evict(self, victim: Request) -> None:
        """Preempt a running request: release its KV pages on the GPU and requeue it.

        recompute (vLLM's default): the KV entries are discarded, so when the request is
            rescheduled it must re-prefill prompt + everything it had generated. Tokens
            already streamed to the user are not lost; the GPU work is, and is charged.
        swap (FastServe-style): the KV entries are copied to host memory and copied back
            on return. No progress is lost; both transfers are charged.

        Which one the engine uses decides whether preemptive policies are viable at all
        -- experiment E6 measures exactly that.
        """
        self.kv_used -= victim.blocks
        self.running.remove(victim)
        if self.cfg.preemption_mode == "swap":
            self.stats.swapped_tokens += victim.kv_tokens()
            self.overhead_s += self.cm.swap_time(victim.kv_tokens())
            victim.swap_out()             # clears victim.blocks
        else:
            self.stats.recomputed_tokens += victim.kv_tokens()
            victim.reset_for_recompute()  # clears victim.blocks
        victim.queued_at = self.now
        self.waiting.append(victim)
        self.stats.preemptions += 1

    def _tokens_to_prefill(self, r: Request) -> int:
        """Tokens this request still needs prefilled for its current admission."""
        return r.prefill_target - r.prefilled

    # ---------- admission ----------

    def _admit(self) -> None:
        """Fill the running batch from the waiting queue, in the policy's order."""
        state = self._state()
        ordered = self.policy.order_waiting(self.waiting, state)

        for req in ordered:
            need = self._blocks_needed(req)
            if need > self.cfg.kv_blocks_total:
                raise RuntimeError(
                    f"request {req.rid} needs {need} KV blocks but the cache only has "
                    f"{self.cfg.kv_blocks_total}; raise kv_blocks_total in config/engine.yaml"
                )

            batch_full = len(self.running) >= self.cfg.max_batch_size
            if batch_full or need > self.kv_free:
                # A preemptive policy may evict less deserving running requests to free a
                # batch slot and/or KV pages. Non-preemptive policies wait for capacity.
                if not (self.policy.preemptive and self.running):
                    break
                if not self._preempt_for(req, need, state):
                    break

            self.waiting.remove(req)
            self.running.append(req)
            req.wait_time += self.now - req.queued_at
            # Reserve for the whole prompt up front, as vLLM does: the engine never
            # admits a prompt it could not hold. A request re-admitted after preemption
            # must prefill its generated tokens again as well.
            self.kv_used += need
            req.blocks = need
            if req.swapped:
                # Copy the parked KV back in. Prefill progress was never lost.
                self.overhead_s += self.cm.swap_time(req.kv_tokens())
                req.swapped = False
            else:
                req.prefill_target = req.kv_tokens()
            if req.start_time is None:
                req.start_time = self.now

    def _preempt_for(self, req: Request, need: int, state: SchedulerState) -> bool:
        """Evict running requests until `req` has a batch slot and `need` KV pages.
        Returns False if it cannot be done.

        Victims are decided before anything is evicted, so a failed attempt costs nothing.
        Only requests the policy ranks strictly *worse* than the candidate are eligible,
        which is what stops SRTF from thrashing two equally-ranked requests forever.
        """
        req_key = self.policy.order_key(req, state)
        candidates = [r for r in self.running if self.policy.order_key(r, state) > req_key]
        candidates.sort(key=lambda r: self.policy.victim_key(r, state), reverse=True)

        freed = self.kv_free
        slots = self.cfg.max_batch_size - len(self.running)
        chosen: list[Request] = []
        for victim in candidates:
            if freed >= need and slots >= 1:
                break
            chosen.append(victim)
            freed += victim.blocks
            slots += 1

        if freed < need or slots < 1:
            return False
        for victim in chosen:
            self._evict(victim)
        return True

    # ---------- one engine iteration ----------

    def _select_prefill_batch(self, budget: int) -> tuple[list[tuple[Request, int]], int]:
        """Pick prefill work up to `budget` tokens, in the policy's order."""
        state = self._state()
        pending = [r for r in self.running if not r.prefill_done]
        pending.sort(key=lambda r: self.policy.order_key(r, state))

        chosen: list[tuple[Request, int]] = []
        used = 0
        for r in pending:
            remaining = self._tokens_to_prefill(r)
            if self.cfg.chunked_prefill:
                take = min(remaining, self.cfg.prefill_chunk_size, budget - used)
            else:
                take = min(remaining, budget - used)
                # A prompt larger than the whole budget still has to run; let it go alone
                # rather than deadlocking the engine.
                if take <= 0 and not chosen:
                    take = remaining
            if take <= 0:
                break
            chosen.append((r, take))
            used += take
            if used >= budget:
                break
        return chosen, used

    def _step(self) -> None:
        """Run exactly one forward pass and advance the clock."""
        decodable = [r for r in self.running if r.prefill_done and not r.done]
        budget = (self.cfg.prefill_chunk_size if self.cfg.chunked_prefill
                  else self.cfg.max_prefill_tokens)
        prefill_batch, prefill_tokens = self._select_prefill_batch(budget)
        # Attention is quadratic per sequence, so the cost model needs each sequence's
        # position, not just the token total.
        segments = [(r.prefilled, take) for r, take in prefill_batch]

        if prefill_batch and self.cfg.chunked_prefill:
            # Sarathi-Serve: fuse the prefill chunk with the ongoing decodes so that
            # decoding never stalls behind a long prompt.
            kind = "hybrid"
            dt = self.cm.hybrid_time(segments, len(decodable))
            advance_decode = True
        elif prefill_batch:
            # vLLM default: a prefill iteration runs alone. Every decoding request is
            # stalled for its duration -- this is the TTFT/TPOT trade-off we measure.
            kind = "prefill"
            dt = self.cm.prefill_time(segments)
            advance_decode = False
        else:
            kind = "decode"
            dt = self.cm.decode_time(len(decodable))
            advance_decode = True

        # Swap transfers queued since the last pass stall this one (no overlap is modelled).
        dt += self.overhead_s
        self.overhead_s = 0.0

        self.now += dt
        self.stats.busy_time += dt
        for r in self.running:
            r.attained_service += dt

        for r, take in prefill_batch:
            r.prefilled += take
            if r.prefill_done:
                # The prefill pass itself produces the first token.
                self._emit_token(r, first=True)

        if advance_decode:
            for r in decodable:
                self._emit_token(r, first=False)

        self._grow_kv()
        self._retire()

        self.stats.iterations += 1
        setattr(self.stats, f"{kind}_iterations",
                getattr(self.stats, f"{kind}_iterations") + 1)
        if self.record_trace:
            self.stats.trace.append(IterationRecord(
                t=self.now, kind=kind, batch_size=len(self.running),
                prefill_tokens=prefill_tokens, kv_blocks_used=self.kv_used,
                n_waiting=len(self.waiting),
            ))
        self.policy.on_iteration_end(self.running, self._state())

    def _emit_token(self, r: Request, *, first: bool) -> None:
        if r.done:
            return
        r.generated += 1
        if first and r.first_token_time is None:
            r.first_token_time = self.now

    def _grow_kv(self) -> None:
        """Charge each running request for the KV pages its new token needs.

        This is where the cache actually fills up, and where a saturated engine starts
        evicting. The victim is chosen by the policy, so an aggressive policy pays for
        its own aggression here.
        """
        for r in list(self.running):
            if r not in self.running:
                continue  # already evicted while making room for an earlier request
            extra = self._blocks_needed(r) - r.blocks
            if extra <= 0:
                continue
            while self.kv_free < extra:
                others = [x for x in self.running if x is not r]
                if not others:
                    break
                self._evict(self.policy.pick_victim(others, self._state()))
            if self.kv_free >= extra:
                self.kv_used += extra
                r.blocks += extra
            else:
                # Nothing left to evict but itself: this request loses its own KV cache
                # and goes back to the queue. This is the pathological case vLLM hits
                # when a single sequence outgrows the entire cache.
                self._evict(r)

    def _retire(self) -> None:
        for r in list(self.running):
            if r.done:
                r.finish_time = self.now
                self.kv_used -= r.blocks
                r.blocks = 0
                self.running.remove(r)

    # ---------- main loop ----------

    def run(self, requests: list[Request]) -> list[Request]:
        pending = sorted(requests, key=lambda r: r.arrival)
        idx = 0
        n = len(pending)

        while idx < n or self.waiting or self.running:
            while idx < n and pending[idx].arrival <= self.now:
                req = pending[idx]
                req.queued_at = req.arrival
                self.waiting.append(req)
                self.policy.on_arrival(req, self._state())
                idx += 1

            if not self.waiting and not self.running:
                # Engine is idle: fast-forward to the next arrival instead of ticking.
                gap = pending[idx].arrival - self.now
                self.now = pending[idx].arrival
                self.stats.idle_time += gap
                continue

            self._admit()

            if not self.running:
                # Nothing could be admitted (cache full of nothing?) -- impossible unless
                # misconfigured, but fail loudly rather than spinning forever.
                raise RuntimeError(
                    f"engine stalled at t={self.now:.3f}s with {len(self.waiting)} waiting "
                    f"and {self.kv_used}/{self.cfg.kv_blocks_total} KV blocks used"
                )

            self._step()

            if self.stats.iterations > self.cfg.max_iterations:
                raise RuntimeError("exceeded max_iterations; is the arrival rate above capacity?")

        return requests


def simulate(
    requests: list[Request],
    policy: Policy,
    *,
    cost_model: CostModel | None = None,
    config: EngineConfig | None = None,
    slo: dict | None = None,
    record_trace: bool = False,
) -> tuple[list[Request], RunStats]:
    """Convenience wrapper: run one policy over one trace."""
    cm = cost_model or CostModel.load()
    cfg = config or EngineConfig.load()
    if slo is None:
        with open(CONFIG_DIR / "workloads.yaml", "r", encoding="utf-8") as fh:
            slo = yaml.safe_load(fh)["slo"]
    eng = Engine(policy, cm, cfg, slo, record_trace=record_trace)
    done = eng.run(requests)
    return done, eng.stats
