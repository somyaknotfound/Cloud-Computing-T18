"""Metric definitions, checked against numbers worked out by hand."""

import math

import numpy as np
import pytest

from src.metrics import aggregate, jains_index, to_frame
from src.workload import Request

SLO = {"interactive": {"ttft_s": 1.0, "tpot_s": 0.1},
       "batch": {"ttft_s": 10.0, "tpot_s": 0.5}}


def finished(rid, arrival, start, first, finish, out, cls="interactive"):
    r = Request(rid=rid, arrival=arrival, prompt_len=50, output_len=out,
                sla_class=cls, predicted_len=out)
    r.start_time, r.first_token_time, r.finish_time = start, first, finish
    r.generated = out
    return r


def test_metrics_match_hand_calculation():
    # arrives at 1.0, starts at 1.5, first token at 2.0, finishes at 4.0, 11 tokens.
    #   queue delay = 0.5     TTFT = 1.0
    #   E2E = 3.0             TPOT = (4.0-2.0)/(11-1) = 0.2
    #   normalized latency = 3.0/11
    df = to_frame([finished(0, 1.0, 1.5, 2.0, 4.0, out=11)])
    row = df.iloc[0]
    assert row["queue_delay"] == pytest.approx(0.5)
    assert row["ttft"] == pytest.approx(1.0)
    assert row["e2e"] == pytest.approx(3.0)
    assert row["tpot"] == pytest.approx(0.2)
    assert row["norm_latency"] == pytest.approx(3.0 / 11)


def test_single_token_request_has_no_tpot():
    """One token means zero inter-token gaps; recording 0.0 would flatter the policy."""
    df = to_frame([finished(0, 0.0, 0.0, 1.0, 1.0, out=1)])
    assert math.isnan(df.iloc[0]["tpot"])

    # ...and it must not be punished for a TPOT it cannot have.
    agg = aggregate(df, SLO)
    assert agg["goodput_frac"] == 1.0


def test_tpot_nan_is_excluded_from_statistics():
    rows = [finished(0, 0.0, 0.0, 0.1, 0.1, out=1),        # NaN TPOT
            finished(1, 0.0, 0.0, 0.1, 1.1, out=11)]       # TPOT = 0.1
    agg = aggregate(to_frame(rows), SLO)
    assert agg["tpot_mean"] == pytest.approx(0.1)


def test_jains_index_bounds():
    assert jains_index(np.array([3.0, 3.0, 3.0])) == pytest.approx(1.0)
    assert jains_index(np.array([1.0, 3.0])) == pytest.approx(0.8)
    # n identical-but-one: index falls towards 1/n as one value dominates.
    skewed = jains_index(np.array([1.0, 1.0, 1.0, 1000.0]))
    assert 0.25 <= skewed < 0.3


def test_goodput_requires_both_ttft_and_tpot():
    rows = [
        finished(0, 0.0, 0.0, 0.5, 1.5, out=11),              # TTFT 0.5, TPOT 0.1 -> ok
        finished(1, 0.0, 0.0, 2.0, 3.0, out=11),              # TTFT 2.0 -> fails TTFT
        finished(2, 0.0, 0.0, 0.5, 10.5, out=11),             # TPOT 1.0 -> fails TPOT
    ]
    agg = aggregate(to_frame(rows), SLO)
    assert agg["goodput_frac"] == pytest.approx(1 / 3)


def test_slo_targets_are_per_class():
    """The same latency passes for a batch job and fails for an interactive one."""
    rows = [finished(0, 0.0, 0.0, 5.0, 6.0, out=11, cls="interactive"),
            finished(1, 0.0, 0.0, 5.0, 6.0, out=11, cls="batch")]
    agg = aggregate(to_frame(rows), SLO)
    assert agg["goodput_interactive"] == 0.0
    assert agg["goodput_batch"] == 1.0


def test_throughput_uses_makespan():
    rows = [finished(0, 0.0, 0.0, 0.1, 5.0, out=100),
            finished(1, 0.0, 0.0, 0.1, 10.0, out=100)]
    agg = aggregate(to_frame(rows), SLO)
    assert agg["makespan_s"] == pytest.approx(10.0)
    assert agg["throughput_tok_s"] == pytest.approx(20.0)
    assert agg["throughput_req_s"] == pytest.approx(0.2)


def test_empty_frame_is_an_error():
    with pytest.raises(ValueError, match="no completed requests"):
        to_frame([])
