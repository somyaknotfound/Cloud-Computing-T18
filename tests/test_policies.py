"""Policy ordering and eviction, checked on hand-built queues where the right answer
can be worked out on paper."""

import pytest

from src.policies import SchedulerState, build
from src.workload import Request

SLO = {"interactive": {"ttft_s": 1.0, "tpot_s": 0.1},
       "batch": {"ttft_s": 10.0, "tpot_s": 0.5}}


def req(rid, arrival, out, cls="interactive", pred=None, prompt=100):
    return Request(rid=rid, arrival=arrival, prompt_len=prompt, output_len=out,
                   sla_class=cls, predicted_len=pred if pred is not None else out)


@pytest.fixture
def state():
    return SchedulerState(now=100.0, kv_blocks_free=50, kv_blocks_total=100,
                          running_count=2, slo=SLO)


@pytest.fixture
def queue():
    #        rid  arrival  out
    return [req(0, 0.0, 500),
            req(1, 1.0, 10),
            req(2, 2.0, 100),
            req(3, 3.0, 50)]


def test_fcfs_is_arrival_order(queue, state):
    p = build("fcfs")
    assert [r.rid for r in p.order_waiting(queue, state)] == [0, 1, 2, 3]


def test_fcfs_evicts_newest(queue, state):
    """vLLM preempts the most recently scheduled request first."""
    assert build("fcfs").pick_victim(queue, state).rid == 3


def test_sjf_orders_by_length_and_evicts_longest(queue, state):
    p = build("sjf-oracle")
    assert [r.rid for r in p.order_waiting(queue, state)] == [1, 3, 2, 0]
    assert p.pick_victim(queue, state).rid == 0


def test_sjf_predicted_follows_the_prediction_not_the_truth(state):
    """The whole point of SJF-Pred: it is wrong exactly when the predictor is wrong."""
    q = [req(0, 0.0, out=10, pred=999),     # short job the predictor thinks is huge
         req(1, 1.0, out=999, pred=10)]     # long job the predictor thinks is tiny
    assert [r.rid for r in build("sjf-pred").order_waiting(q, state)] == [1, 0]
    assert [r.rid for r in build("sjf-oracle").order_waiting(q, state)] == [0, 1]


def test_srtf_uses_remaining_not_total(state):
    """A long request that is nearly done must outrank a shorter one that has not started."""
    nearly_done = req(0, 0.0, out=500)
    nearly_done.generated = 495           # 5 tokens left
    fresh = req(1, 1.0, out=50)           # 50 tokens left
    assert [r.rid for r in build("srtf").order_waiting([fresh, nearly_done], state)] == [0, 1]


def test_priority_puts_interactive_first(state):
    q = [req(0, 0.0, 100, cls="batch"),
         req(1, 5.0, 100, cls="interactive"),
         req(2, 1.0, 100, cls="batch"),
         req(3, 6.0, 100, cls="interactive")]
    p = build("priority")
    assert [r.rid for r in p.order_waiting(q, state)] == [1, 3, 0, 2]
    assert p.pick_victim(q, state).rid == 2, "evict batch with the latest deadline"


def test_mlfq_las_starts_everyone_at_the_top(state):
    """MLFQ-LAS must not peek at the prediction: a request predicted to be huge still
    starts at level 0 and has to *prove* it is long by consuming service."""
    p = build("mlfq", levels=8, base_quantum_tokens=16, starvation_s=1e9)
    tiny, huge = req(0, 0.0, 8), req(1, 0.0, 2000)
    p.on_arrival(tiny, state)
    p.on_arrival(huge, state)
    assert p.level[tiny.rid] == p.level[huge.rid] == 0


def test_mlfq_pred_skip_join_places_by_predicted_size(state):
    p = build("mlfq-pred", levels=8, base_quantum_tokens=16, starvation_s=1e9)
    tiny, huge = req(0, 0.0, 8), req(1, 0.0, 2000)
    p.on_arrival(tiny, state)
    p.on_arrival(huge, state)
    assert p.level[tiny.rid] == 0
    assert p.level[huge.rid] > p.level[tiny.rid]


def test_mlfq_demotes_on_service(state):
    p = build("mlfq", levels=8, base_quantum_tokens=16, starvation_s=1e9)
    r = req(0, 0.0, 5000)
    p.level[r.rid] = 0
    r.generated = 100
    p.on_iteration_end([r], state)
    assert p.level[r.rid] > 0, "a request that keeps consuming service must be demoted"


def test_mlfq_ageing_rescues_a_starved_request():
    """The anti-starvation valve: waiting past the threshold returns you to level 0."""
    p = build("mlfq", levels=8, base_quantum_tokens=16, starvation_s=10.0)
    starved = req(0, 0.0, 5000)
    p.level[starved.rid] = 7

    fresh = SchedulerState(now=5.0, kv_blocks_free=1, kv_blocks_total=1,
                           running_count=0, slo=SLO)
    aged = SchedulerState(now=50.0, kv_blocks_free=1, kv_blocks_total=1,
                          running_count=0, slo=SLO)
    assert p.order_key(starved, fresh)[0] == 7
    assert p.order_key(starved, aged)[0] == 0


def test_preemptive_flags():
    """SRTF and both MLFQs only make sense if they can evict; the rest never do it by choice."""
    for name in ("srtf", "mlfq", "mlfq-pred"):
        assert build(name).preemptive, name
    for name in ("fcfs", "sjf-oracle", "sjf-pred", "priority"):
        assert not build(name).preemptive, name


def test_unknown_policy_raises():
    with pytest.raises(KeyError, match="unknown policy"):
        build("shortest-elephant-first")
