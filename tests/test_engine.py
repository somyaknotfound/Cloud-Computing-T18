"""Engine invariants.

These are the tests that make the simulator trustworthy. If any of them fail, every
number in the report is meaningless -- so they check physical impossibilities rather
than specific values: no request may vanish, no token may be conjured, the KV cache
may never hold more than it has, and time may not run backwards.
"""

import pandas as pd
import pytest

from src.engine import Engine, EngineConfig, simulate
from src.metrics import to_frame
from src.policies import ALL_POLICIES, build
from src.workload import Request, generate


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_every_request_completes_exactly(policy_name, cost_model, config, slo):
    reqs = generate("default", num_requests=200, arrival_rate=8.0)
    done, _ = simulate(reqs, build(policy_name), cost_model=cost_model,
                       config=config, slo=slo)

    assert len(done) == 200
    for r in done:
        assert r.generated == r.output_len, f"{r.rid} generated {r.generated}/{r.output_len}"
        assert r.finish_time is not None
        assert r.first_token_time is not None
        # Causality: arrive, then start, then first token, then finish.
        assert r.arrival <= r.start_time <= r.first_token_time <= r.finish_time


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_kv_cache_never_oversubscribed(policy_name, cost_model, tight_config, slo):
    """The cache is the hard physical limit. Exceeding it would mean the simulator is
    reporting throughput the hardware could not actually deliver."""
    reqs = generate("default", num_requests=80, arrival_rate=8.0)
    for r in reqs:                      # keep prompts inside the cramped test cache
        r.prompt_len = min(r.prompt_len, 128)
        r.output_len = min(r.output_len, 120)
        r.predicted_len = r.output_len

    _, stats = simulate(reqs, build(policy_name), cost_model=cost_model,
                        config=tight_config, slo=slo, record_trace=True)

    assert stats.trace, "trace was not recorded"
    for rec in stats.trace:
        assert 0 <= rec.kv_blocks_used <= tight_config.kv_blocks_total
        assert rec.batch_size <= tight_config.max_batch_size


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_time_is_monotonic(policy_name, cost_model, config, slo):
    reqs = generate("default", num_requests=120, arrival_rate=8.0)
    _, stats = simulate(reqs, build(policy_name), cost_model=cost_model,
                        config=config, slo=slo, record_trace=True)
    times = [rec.t for rec in stats.trace]
    assert times == sorted(times)
    assert all(t > 0 for t in times)


def test_kv_accounting_balances(cost_model, tight_config, slo):
    """Every block reserved must be released: the cache is empty once the queue drains."""
    reqs = generate("default", num_requests=60, arrival_rate=8.0)
    for r in reqs:
        r.prompt_len = min(r.prompt_len, 128)
        r.output_len = min(r.output_len, 100)
        r.predicted_len = r.output_len

    eng = Engine(build("fcfs"), cost_model, tight_config, slo)
    eng.run(reqs)

    assert eng.kv_used == 0
    assert eng.running == []
    assert eng.waiting == []
    assert all(r.blocks == 0 for r in reqs)


def test_preemption_charges_recompute(cost_model, tight_config, slo):
    """A preempted request must redo the prompt AND the tokens it had already generated.

    Preemption has to be expensive, otherwise an aggressive policy would look free.
    """
    reqs = generate("default", num_requests=60, arrival_rate=12.0)
    for r in reqs:
        r.prompt_len = min(r.prompt_len, 160)
        r.output_len = max(60, min(r.output_len, 200))
        r.predicted_len = r.output_len

    _, stats = simulate(reqs, build("srtf"), cost_model=cost_model,
                        config=tight_config, slo=slo)

    assert stats.preemptions > 0, "cramped cache should have forced preemptions"
    assert stats.recomputed_tokens >= stats.preemptions
    assert sum(r.preemptions for r in reqs) == stats.preemptions


def _two_request_trace(long_prompt=32):
    """A long request alone in a one-slot engine, then a short one arrives behind it."""
    long_ = Request(rid=0, arrival=0.0, prompt_len=long_prompt, output_len=400,
                    sla_class="interactive", predicted_len=400)
    short = Request(rid=1, arrival=0.5, prompt_len=32, output_len=5,
                    sla_class="interactive", predicted_len=5)
    return [long_, short]


@pytest.mark.parametrize("mode", ["recompute", "swap"])
def test_preemptive_policy_evicts_for_a_batch_slot(mode, cost_model, slo):
    """With the batch full but memory to spare, SRTF must still evict the long request
    for the short one -- otherwise "preemptive" would only ever mean "under KV pressure"."""
    cfg = EngineConfig.load(max_batch_size=1, preemption_mode=mode)
    long_, short = _two_request_trace()
    simulate([long_, short], build("srtf"), cost_model=cost_model, config=cfg, slo=slo)

    assert long_.preemptions == 1
    assert short.finish_time < long_.finish_time
    # The evicted request's second stint in the queue counts as waiting.
    assert long_.wait_time > long_.start_time - long_.arrival


def test_non_preemptive_policy_waits_for_the_slot(cost_model, slo):
    cfg = EngineConfig.load(max_batch_size=1)
    long_, short = _two_request_trace()
    simulate([long_, short], build("sjf-oracle"), cost_model=cost_model, config=cfg, slo=slo)
    assert long_.preemptions == 0
    assert short.start_time >= long_.finish_time


def test_swap_preemption_keeps_progress(cost_model, slo):
    """Swap mode must never re-prefill: preemptions happen, nothing is recomputed, and
    every token still comes out exactly once."""
    cfg = EngineConfig.load(max_batch_size=1, preemption_mode="swap")
    long_, short = _two_request_trace()
    _, stats = simulate([long_, short], build("srtf"), cost_model=cost_model,
                        config=cfg, slo=slo)
    assert stats.preemptions == 1
    assert stats.recomputed_tokens == 0
    assert stats.swapped_tokens > 0
    assert long_.generated == long_.output_len and short.generated == short.output_len


def test_swap_is_cheaper_than_recompute_for_a_large_victim(cost_model, slo):
    """E6 rests on this: a victim holding a lot of KV finishes sooner if it was swapped.

    Not true for a tiny context: re-prefilling 70 tokens costs about one decode step and
    also yields the next token, so recompute can win there. Hence the 1500-token prompt.
    """
    finish = {}
    for mode in ("recompute", "swap"):
        long_, short = _two_request_trace(long_prompt=1500)
        simulate([long_, short], build("srtf"), cost_model=cost_model,
                 config=EngineConfig.load(max_batch_size=1, preemption_mode=mode), slo=slo)
        finish[mode] = long_.finish_time
    assert finish["swap"] < finish["recompute"]


def test_unknown_preemption_mode_is_rejected(cost_model, slo):
    with pytest.raises(ValueError, match="preemption_mode"):
        simulate(generate("default", num_requests=5), build("fcfs"), cost_model=cost_model,
                 config=EngineConfig.load(preemption_mode="teleport"), slo=slo)


@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_simulation_is_deterministic(policy_name, cost_model, config, slo):
    """Same trace, same policy, same config -> bit-identical per-request results. This is
    what lets `python -m src.runner --verify` demand byte-identical CSVs."""
    frames = []
    for _ in range(2):
        done, _ = simulate(generate("default", num_requests=150, arrival_rate=10.0),
                           build(policy_name), cost_model=cost_model, config=config, slo=slo)
        frames.append(to_frame(done))
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_idle_engine_fast_forwards(cost_model, config, slo):
    """With arrivals far apart the engine must jump to the next arrival, not tick."""
    reqs = generate("default", num_requests=10, arrival_rate=0.05)
    _, stats = simulate(reqs, build("fcfs"), cost_model=cost_model, config=config, slo=slo)
    assert stats.idle_time > 0
    assert stats.iterations < 5000


def test_oversized_prompt_is_rejected_loudly(cost_model, slo):
    """A prompt bigger than the whole cache is a config error, not a silent hang."""
    cfg = EngineConfig.load(kv_blocks_total=4, block_size=16)   # 64 tokens of capacity
    reqs = generate("default", num_requests=3, arrival_rate=1.0)
    reqs[0].prompt_len = 5000
    with pytest.raises(RuntimeError, match="KV blocks"):
        simulate(reqs, build("fcfs"), cost_model=cost_model, config=cfg, slo=slo)


def test_chunked_prefill_produces_same_output(cost_model, slo):
    """Chunking changes *when* tokens come out, never *how many*."""
    base = EngineConfig.load(chunked_prefill=False)
    chunked = EngineConfig.load(chunked_prefill=True, prefill_chunk_size=128)

    a, _ = simulate(generate("default", num_requests=100, arrival_rate=8.0),
                    build("fcfs"), cost_model=cost_model, config=base, slo=slo)
    b, _ = simulate(generate("default", num_requests=100, arrival_rate=8.0),
                    build("fcfs"), cost_model=cost_model, config=chunked, slo=slo)

    assert [r.output_len for r in a] == [r.output_len for r in b]
    assert all(r.generated == r.output_len for r in b)
