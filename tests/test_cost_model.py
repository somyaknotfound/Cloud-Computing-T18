"""The cost model and its fitter: the only link between the simulator and a real GPU."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cost_model import CostModel

CALIB = Path(__file__).resolve().parent.parent / "calibration"
_spec = importlib.util.spec_from_file_location("fit_cost_model", CALIB / "fit_cost_model.py")
fitter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fitter)

PLANTED = dict(prefill_floor_s=0.030, prefill_per_token_s=6e-5, prefill_per_token2_s=8e-8,
               decode_base_s=0.025, decode_per_seq_s=2e-4,
               swap_base_s=3e-4, swap_per_token_s=1e-6)


def test_prefill_never_goes_below_the_floor(cost_model):
    """The linear fit we first tried returned negative times for short prompts."""
    for n in range(1, 5000, 37):
        assert cost_model.prefill_time([(0, n)]) >= cost_model.prefill_floor_s > 0


def test_chunking_does_not_change_total_prefill_compute(cost_model):
    """A chunk attends to everything before it, so chunks must add up to the whole prompt."""
    whole = cost_model.prefill_compute([(0, 2000)])
    chunks = sum(cost_model.prefill_compute([(s, 64)]) for s in range(0, 1984, 64)) \
        + cost_model.prefill_compute([(1984, 16)])
    assert chunks == pytest.approx(whole, rel=1e-12)


def test_attention_is_quadratic_per_sequence(cost_model):
    """Eight 256-token prompts in one pass must cost less than one 2048-token prompt."""
    batched = cost_model.prefill_compute([(0, 256)] * 8)
    single = cost_model.prefill_compute([(0, 2048)])
    assert batched < single


def _synthetic(noise=0.01, outlier_at=4, seed=0):
    rng = np.random.default_rng(seed)
    cm = CostModel(**PLANTED)
    rows = []
    for n in (32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048):
        for _ in range(7):
            rows.append(("prefill", 1, n, cm.prefill_time([(0, n)]) * (1 + rng.normal(0, noise))))
    for b in (1, 2, 4, 8, 12, 16, 24, 32):
        slow = 1.5 if b == outlier_at else 1.0     # one transient slow setting, as on the T4
        for _ in range(7):
            rows.append(("decode", b, 1, cm.decode_time(b) * slow * (1 + rng.normal(0, noise))))
    for k in (128, 256, 512, 1024, 2048):
        for _ in range(7):
            rows.append(("swap", 1, k, cm.swap_time(k) * (1 + rng.normal(0, noise))))
    return pd.DataFrame(rows, columns=["phase", "batch_size", "tokens", "latency_s"])


def test_fitter_recovers_planted_constants_despite_an_outlier():
    fitted = fitter.fit(_synthetic())
    for key, planted in PLANTED.items():
        assert fitted[key] == pytest.approx(planted, rel=0.10), key
    assert fitted["decode_per_seq_s"] > 0, "one slow batch size must not flip the slope"
    for name, q in fitted["fit_quality"].items():
        assert q["median_rel_err"] < fitter.MAX_MEDIAN_ERR, name


def test_committed_cost_model_is_what_the_committed_csv_fits_to():
    """Nobody hand-edited the constants: refitting the raw measurements reproduces them."""
    committed = json.loads((CALIB / "cost_model.json").read_text(encoding="utf-8"))
    if not committed["source"].startswith("measured"):
        pytest.skip("cost model is not measured yet")
    refit = fitter.fit(pd.read_csv(CALIB / "calibration_t4.csv"))
    for key in PLANTED:
        assert refit[key] == committed[key], key
    for name, q in committed["fit_quality"].items():
        assert q["median_rel_err"] <= fitter.MAX_MEDIAN_ERR, name
