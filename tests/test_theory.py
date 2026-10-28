"""The theory sanity gate.

A simulator can be internally consistent and still be modelling the wrong universe.
These tests check that it reproduces what scheduling theory, or the mechanism it is built
on, says *must* hold. They are the argument that the numbers in the report mean something.

Each robust claim is asserted on three traces, not one: seed 42 (the one every figure
uses) plus seeds 4 and 9, which are the seeds where claims that only held on trace #42
broke. A claim belongs here only if it held on every one of the eleven traces E7 checks.

Two things that look like theory and are deliberately NOT asserted:
- "All policies reach the same throughput." Under continuous batching the token rate
  depends on how full the batch is, so a policy that leaves the longest jobs for the end
  (SJF) drains with a near-empty batch. Measured spread: up to ~10% between non-preemptive
  policies. What theory does require is that no policy beats the engine's ceiling.
- "MLFQ-LAS beats FCFS on mean latency." LAS only beats FCFS when job sizes have a
  decreasing hazard rate; ours are lognormal, and E7 finds it on 9 of 11 traces. The report
  says so instead of the test pretending otherwise.
"""

from functools import lru_cache

import pytest

from src.cost_model import CostModel
from src.engine import EngineConfig, simulate
from src.metrics import aggregate, to_frame
from src.policies import ALL_POLICIES, build
from src.workload import generate, load_workload_config

SEEDS = (42, 4, 9)
MODES = ("recompute", "swap")
PREEMPTIVE = ("srtf", "mlfq", "mlfq-pred")


@lru_cache(maxsize=None)
def run(policy, mode="recompute", seed=42, sigma=0.0, rate=10.0, n=600):
    """Aggregate metrics for one point; cached because several tests share points."""
    slo = load_workload_config()["slo"]
    reqs = generate("default", num_requests=n, arrival_rate=rate, seed=seed,
                    predictor_error=sigma)
    done, stats = simulate(reqs, build(policy), cost_model=CostModel.load(),
                           config=EngineConfig.load(preemption_mode=mode), slo=slo)
    row = aggregate(to_frame(done), slo)
    row["recomputed_tokens"] = stats.recomputed_tokens
    return row


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_sjf_beats_fcfs_on_mean_latency(mode, seed):
    """SJF minimises mean completion time. If the simulator disagrees, it is wrong."""
    fcfs, sjf = run("fcfs", mode, seed), run("sjf-oracle", mode, seed)
    # The gap should be large, not marginal: this is the headline result.
    assert sjf["norm_latency_mean"] < 0.5 * fcfs["norm_latency_mean"]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_sjf_starves_someone_and_fcfs_does_not(mode, seed):
    """SJF buys its average by making the worst case worse. That is the trade-off the
    whole project exists to quantify."""
    assert run("sjf-oracle", mode, seed)["max_wait"] > run("fcfs", mode, seed)["max_wait"]


@pytest.mark.parametrize("mode", MODES)
def test_scheduling_cannot_create_capacity(mode):
    """No ordering of the work can beat a full batch decoding back to back."""
    cm, cfg = CostModel.load(), EngineConfig.load()
    ceiling = cfg.max_batch_size / cm.decode_time(cfg.max_batch_size)
    for p in ALL_POLICIES:
        assert run(p, mode)["throughput_tok_s"] < ceiling, p


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("policy", PREEMPTIVE)
def test_recompute_preemption_wastes_capacity(policy, seed):
    """Recompute throws away the victim's KV and pays to rebuild it, so a policy that
    preempts often must lose throughput relative to the same policy with cheap swaps."""
    rec, swp = run(policy, "recompute", seed), run(policy, "swap", seed)
    assert rec["recomputed_tokens"] > 0
    assert rec["throughput_tok_s"] < swp["throughput_tok_s"]


@pytest.mark.parametrize("mode", MODES)
def test_perfect_predictor_reproduces_the_oracle(mode):
    """SJF-Pred with zero prediction error must be indistinguishable from SJF-Oracle.
    This is what makes the E3 sweep interpretable: at sigma=0 the two curves must meet."""
    oracle, perfect = run("sjf-oracle", mode), run("sjf-pred", mode)
    assert perfect["norm_latency_mean"] == pytest.approx(oracle["norm_latency_mean"], rel=1e-9)


def test_bad_predictions_erode_the_sjf_advantage():
    """A predictor is only worth having if it is good enough. With enough noise, SJF-Pred
    must lose its edge -- this is the mechanism experiment E3 measures."""
    good = run("sjf-pred", sigma=0.1)
    bad = run("sjf-pred", sigma=1.5)
    assert bad["norm_latency_mean"] > good["norm_latency_mean"]


@pytest.mark.parametrize("mode", MODES)
def test_mlfq_ignores_the_predictor(mode):
    """MLFQ-LAS claims to need no predictor. Then a perfect predictor and a useless one
    must produce bit-identical results -- not similar ones."""
    perfect = run("mlfq", mode, sigma=0.0)
    useless = run("mlfq", mode, sigma=8.0)
    assert perfect == useless


@pytest.mark.parametrize("seed", SEEDS)
def test_mlfq_las_collapses_under_recompute(seed):
    """LAS keeps evicting long requests for fresh ones. With recompute, every eviction
    re-prefills the victim and stalls the batch, which is worse than not scheduling."""
    assert run("mlfq", "recompute", seed)["norm_latency_mean"] > \
        run("fcfs", "recompute", seed)["norm_latency_mean"]


@pytest.mark.parametrize("seed", SEEDS)
def test_mlfq_las_with_cheap_preemption(seed):
    """With swaps, LAS does what it is for: a fresh request is served at once (TTFT), and
    ageing keeps the worst case well below SJF's -- all without a predictor."""
    mlfq, fcfs, sjf = (run(p, "swap", seed) for p in ("mlfq", "fcfs", "sjf-oracle"))
    assert mlfq["ttft_p50"] < fcfs["ttft_p50"]
    assert mlfq["goodput_frac"] > fcfs["goodput_frac"]
    assert mlfq["max_wait"] < sjf["max_wait"]


@pytest.mark.parametrize("seed", SEEDS)
def test_cheap_preemption_closes_srtfs_gap_to_sjf(seed):
    """The SRPT intuition says a preemptive SJF should do at least as well as SJF -- if
    setting a job aside is cheap. Under recompute SRTF loses to SJF on every trace; with
    swaps it must draw level (within 5%). "SRTF strictly beats SJF" held on only 9 of 11
    traces on the measured T4 constants, so it is not asserted."""
    def ratio(mode):
        return run("srtf", mode, seed)["norm_latency_mean"] / \
            run("sjf-oracle", mode, seed)["norm_latency_mean"]
    assert ratio("recompute") > 1.0
    assert ratio("swap") < 1.05
    assert ratio("swap") < ratio("recompute")


def test_priority_protects_interactive_traffic():
    """The point of a priority policy is that the favoured class does better than it
    would under FCFS -- and the other class pays for it."""
    fcfs, prio = run("fcfs"), run("priority")
    assert prio["norm_latency_mean_interactive"] < fcfs["norm_latency_mean_interactive"]
    assert prio["norm_latency_mean_batch"] > fcfs["norm_latency_mean_batch"]


@pytest.mark.parametrize("mode", MODES)
def test_below_saturation_policy_choice_stops_mattering(mode):
    """The control experiment. With a near-empty queue the scheduler never has a decision
    to make, so all policies must converge. A simulator that showed big differences at
    low load would be manufacturing them."""
    fcfs = run("fcfs", mode, rate=2.0, n=300)["norm_latency_mean"]
    for p in ALL_POLICIES:
        ratio = run(p, mode, rate=2.0, n=300)["norm_latency_mean"] / fcfs
        assert 0.8 < ratio < 1.2, f"{p} should converge at low load, ratio={ratio:.2f}"
