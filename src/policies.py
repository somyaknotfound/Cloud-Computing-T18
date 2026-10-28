"""The seven scheduling policies under test.

Owned by: Chirs Tony (241CS222)

The central simplification of this project: **a scheduling policy is just two sort keys.**

  order_key(req)   -> requests waiting to be admitted are served in increasing order of this
  victim_key(req)  -> when the KV cache is full, the RUNNING request with the LARGEST
                      value of this is the one evicted

Everything else -- batching, the KV cache, preemption mechanics, timing -- lives in
engine.py and is identical for every policy. That is what makes this a fair comparison:
the only thing that changes between runs is these two functions.

A policy may only read fields a real serving system could actually observe on arrival:
`arrival`, `prompt_len`, `sla_class`, `predicted_len`, and its own accumulated state.
It must NOT read `output_len` or `remaining` -- those are ground truth. The single
exception is SJFOracle, which reads them deliberately to establish an unreachable
lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .workload import Request


@dataclass
class SchedulerState:
    """Read-only snapshot of the engine, handed to the policy on every iteration."""

    now: float
    kv_blocks_free: int
    kv_blocks_total: int
    running_count: int
    slo: dict


class Policy:
    """Base class. Subclasses override the two keys."""

    name = "base"
    #: may evict a running request purely to admit a more deserving waiting one
    #: (as opposed to only evicting when the KV cache has genuinely run out)
    preemptive = False

    def on_arrival(self, req: Request, state: SchedulerState) -> None:
        """Hook for policies that keep per-request bookkeeping (MLFQ uses it)."""

    def on_iteration_end(self, running: list[Request], state: SchedulerState) -> None:
        """Hook called after every engine iteration (MLFQ demotes here)."""

    def order_key(self, req: Request, state: SchedulerState):
        raise NotImplementedError

    def victim_key(self, req: Request, state: SchedulerState):
        """Largest value is evicted first. Default: evict the newest arrival, as vLLM does."""
        return req.arrival

    def order_waiting(self, waiting: list[Request], state: SchedulerState) -> list[Request]:
        return sorted(waiting, key=lambda r: self.order_key(r, state))

    def pick_victim(self, running: list[Request], state: SchedulerState) -> Request:
        return max(running, key=lambda r: self.victim_key(r, state))


class FCFS(Policy):
    """First come, first served -- what vLLM and Orca actually ship as the default.

    No prediction, no starvation, trivially fair in arrival order. Its weakness is
    head-of-line blocking: one 2000-token generation parked in the batch delays every
    short request queued behind it.
    """

    name = "FCFS"

    def order_key(self, req, state):
        return req.arrival


class SJFOracle(Policy):
    """Shortest job first, using the TRUE output length.

    Physically unrealisable -- nobody knows how long an answer will be before generating
    it. It is here as the theoretical lower bound on mean latency (SJF is provably optimal
    for mean completion time on a single server), so we can measure how much of that
    headroom a real predictor actually captures.
    """

    name = "SJF-Oracle"

    def order_key(self, req, state):
        return (req.output_len, req.arrival)

    def victim_key(self, req, state):
        return req.remaining  # evict whoever has the most work left


class SJFPredicted(Policy):
    """Shortest job first using a *predicted* output length.

    This is the realistic version of SJF and the one production proposals actually use
    (length predictors, learning-to-rank). The prediction error is a swept parameter of
    the workload, so experiment E3 can answer the real question: how accurate must the
    predictor be before SJF beats plain FCFS?
    """

    name = "SJF-Pred"

    def order_key(self, req, state):
        return (req.predicted_len, req.arrival)

    def victim_key(self, req, state):
        return req.predicted_remaining


class SRTF(Policy):
    """Shortest remaining time first -- preemptive SJF over predicted remaining tokens.

    Unlike SJF it re-evaluates continuously, so a long request that has nearly finished
    outranks a fresh medium one. Best mean latency of the non-oracle policies, and the
    worst starvation: a long request can be evicted repeatedly and never make progress.
    We measure exactly that with max_queue_delay and Jain's index.
    """

    name = "SRTF"
    preemptive = True

    def order_key(self, req, state):
        return (req.predicted_remaining, req.arrival)

    def victim_key(self, req, state):
        return req.predicted_remaining


class MLFQ(Policy):
    """Multi-level feedback queue over least attained service -- needs no predictor.

    The idea: we cannot know a request's length, so *learn* it by watching how much
    service it consumes. Every request starts at level 0 and is demoted as it proves
    itself long. Short requests finish while still in a high queue, so they get
    SJF-like latency without any length prediction at all. This class reads nothing
    but arrival time and tokens already generated; `test_mlfq_ignores_the_predictor`
    holds it to that.

    It must be **preemptive**. Under continuous batching, every request that is waiting
    has received no service yet, so all of them sit at level 0 and admission alone is
    just FCFS. The feedback only bites when a demoted request *in the batch* can be
    evicted for a fresh one, as in FastServe. Eviction is recompute, so it is charged.

    Ageing: anything older than `starvation_s` (measured from arrival) is treated as
    level 0 again. This is the explicit anti-starvation valve that SRTF lacks.

    Service is accounted in generated tokens rather than GPU-seconds. Under continuous
    batching a decode step costs every sequence in the batch the same, so tokens are
    proportional to service received and are far easier to reason about.

    The defaults matter more than they look. The level quanta are 16, 32, 64 ... 2048
    tokens, chosen to span the actual output-length range of the workload -- too few
    levels and every request lands in the same bucket, which silently degenerates into
    FCFS. `starvation_s` has the same failure mode from the other direction: set it below
    the delay requests actually experience and *everything* is permanently promoted to
    level 0, which is again just FCFS. Experiment E4 sweeps it.
    """

    name = "MLFQ-LAS"
    preemptive = True

    def __init__(self, levels: int = 8, base_quantum_tokens: int = 16, starvation_s: float = 30.0):
        self.levels = levels
        self.base_quantum = base_quantum_tokens
        self.starvation_s = starvation_s
        self.level: dict[int, int] = {}

    def _quantum(self, level: int) -> int:
        """Level k tolerates base * 2^k tokens of service before demotion."""
        return self.base_quantum * (2 ** level)

    def on_arrival(self, req, state):
        self.level[req.rid] = 0

    def on_iteration_end(self, running, state):
        for r in running:
            lvl = self.level.get(r.rid, 0)
            # Demote once this request has consumed more than its level's quantum.
            while lvl < self.levels - 1 and r.generated > self._quantum(lvl):
                lvl += 1
            self.level[r.rid] = lvl

    def order_key(self, req, state):
        lvl = self.level.get(req.rid, 0)
        # Ageing: a request starved past the threshold jumps back to the top.
        if state.now - req.arrival > self.starvation_s:
            lvl = 0
        return (lvl, req.arrival)

    def victim_key(self, req, state):
        return (self.level.get(req.rid, 0), req.arrival)


class MLFQPredicted(MLFQ):
    """MLFQ-LAS plus skip-join driven by a length predictor.

    Identical to MLFQ-LAS except where a request *starts*: at the level its predicted
    output length would be demoted to anyway, so an obviously long request never holds a
    top-priority slot. Comparing the two isolates exactly what a predictor adds to MLFQ.

    FastServe's own skip-join keys on the prefill time of the first iteration, i.e. the
    prompt length. Our quanta are in output tokens, so the predicted output length is the
    quantity that lands on the same scale.
    """

    name = "MLFQ-Pred"

    def on_arrival(self, req, state):
        if req.predicted_len <= self.base_quantum:
            start = 0
        else:
            start = min(self.levels - 1,
                        int(math.log2(req.predicted_len / self.base_quantum)) + 1)
        self.level[req.rid] = start


class PrioritySLO(Policy):
    """Strict class priority, earliest-deadline-first within a class.

    The multi-tenant cloud case: interactive chat traffic and bulk batch jobs share one
    GPU and have different SLOs. Interactive always outranks batch; ties break on the
    TTFT deadline the request must hit (arrival + its class TTFT target), which is
    classic EDF adapted to LLM serving. When the cache fills, batch work is evicted first.
    """

    name = "Priority-SLO"

    CLASS_RANK = {"interactive": 0, "batch": 1}

    def _deadline(self, req, state):
        return req.arrival + state.slo[req.sla_class]["ttft_s"]

    def order_key(self, req, state):
        return (self.CLASS_RANK[req.sla_class], self._deadline(req, state))

    def victim_key(self, req, state):
        # Evict batch before interactive; within a class, evict the one with the most
        # slack (the latest deadline).
        return (self.CLASS_RANK[req.sla_class], self._deadline(req, state))


REGISTRY = {
    "fcfs": FCFS,
    "sjf-oracle": SJFOracle,
    "sjf-pred": SJFPredicted,
    "srtf": SRTF,
    "mlfq": MLFQ,
    "mlfq-pred": MLFQPredicted,
    "priority": PrioritySLO,
}

ALL_POLICIES = list(REGISTRY)


def build(name: str, **kwargs) -> Policy:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}; have {ALL_POLICIES}")
    return REGISTRY[key](**kwargs)
