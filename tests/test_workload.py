"""Workload generation: reproducibility and the oracle/predictor separation."""

import numpy as np
import pytest

from src.workload import Request, generate, summarize


def test_same_seed_same_trace():
    a = generate("default", num_requests=100, seed=7)
    b = generate("default", num_requests=100, seed=7)
    assert [(r.arrival, r.prompt_len, r.output_len) for r in a] == \
           [(r.arrival, r.prompt_len, r.output_len) for r in b]


def test_different_seed_different_trace():
    a = generate("default", num_requests=100, seed=7)
    b = generate("default", num_requests=100, seed=8)
    assert [r.output_len for r in a] != [r.output_len for r in b]


def test_zero_error_predictor_is_an_oracle():
    reqs = generate("default", num_requests=200, predictor_error=0.0)
    assert all(r.predicted_len == r.output_len for r in reqs)


@pytest.mark.parametrize("sigma", [0.25, 0.5, 1.0])
def test_predictor_error_scales_monotonically(sigma):
    """Bigger sigma must mean a worse predictor -- this is the x-axis of experiment E3."""
    reqs = generate("default", num_requests=800, predictor_error=sigma, seed=3)
    rel = np.array([abs(r.predicted_len - r.output_len) / r.output_len for r in reqs])
    assert 0 < np.median(rel) < 3.0
    assert all(r.predicted_len >= 1 for r in reqs)

    smaller = generate("default", num_requests=800, predictor_error=sigma / 2, seed=3)
    rel_small = np.array([abs(r.predicted_len - r.output_len) / r.output_len
                          for r in smaller])
    assert np.median(rel_small) < np.median(rel)


def test_lengths_respect_configured_bounds():
    reqs = generate("default", num_requests=1000)
    assert all(8 <= r.prompt_len <= 4096 for r in reqs)
    assert all(1 <= r.output_len <= 2048 for r in reqs)


def test_arrivals_are_ordered_and_match_rate():
    reqs = generate("default", num_requests=2000, arrival_rate=10.0, seed=11)
    arrivals = [r.arrival for r in reqs]
    assert arrivals == sorted(arrivals)
    assert summarize(reqs)["arrival_rate_actual"] == pytest.approx(10.0, rel=0.1)


def test_bimodal_workload_has_two_modes():
    reqs = generate("bimodal", num_requests=2000)
    outs = np.array([r.output_len for r in reqs])
    # 80% short / 20% long by construction: the top decile must dwarf the median.
    assert np.percentile(outs, 95) > 8 * np.median(outs)


def test_request_state_helpers():
    r = Request(rid=0, arrival=0.0, prompt_len=100, output_len=10,
                sla_class="interactive", predicted_len=10)
    r.prefill_target = 100
    assert not r.prefill_done and not r.done
    r.prefilled = 100
    assert r.prefill_done
    r.generated = 4
    assert r.kv_tokens() == 104 and r.remaining == 6

    r.reset_for_recompute()
    assert r.prefilled == 0 and r.blocks == 0 and r.preemptions == 1
    assert r.kv_tokens() == 104, "generated tokens still need KV space after recompute"
