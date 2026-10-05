"""Workstream 6: calibrated probabilities + abstention.

Probabilities stay MLE (no hidden smoothing) but carry Wilson 95%
intervals and evidence counts; predict_next abstains when confidence
or evidence is too thin. Variable-order backoff must beat order-1 on
held-out higher-order data.
"""
import random

from nexora import Nexora
from nexora.prediction.calibration import calibrate, wilson_interval
from nexora.prediction.context import build_context_model, predict_with_context
from nexora.prediction.markov import build_transition_matrix, predict_next


def test_wilson_exact_spots():
    lo, hi = wilson_interval(5, 5)
    assert abs(lo - 0.5655) < 0.005 and hi == 1.0
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = wilson_interval(1, 2)
    assert abs(lo - 0.0945) < 0.005 and abs(hi - 0.9055) < 0.005
    lo, hi = wilson_interval(100, 100)
    assert lo > 0.95 and hi == 1.0  # tight on strong evidence
    lo, hi = wilson_interval(1, 1)
    assert lo < 0.3  # wide on thin evidence


def test_calibrate_shape():
    c = calibrate(5, 5)
    assert c["probability"] == 1.0 and c["count"] == 5 and c["total"] == 5
    assert len(c["ci95"]) == 2 and c["ci95"][0] <= 1.0 <= c["ci95"][1]
    c0 = calibrate(0, 0)
    assert c0["probability"] == 0.0 and c0["ci95"] == [0.0, 1.0]


def test_markov_entries_calibrated_mle_unchanged():
    m = build_transition_matrix(["A", "B", "A", "B", "A", "C"])
    top = predict_next("A", m)[0]
    assert top["next"] == "B"
    assert top["probability"] == 2 / 3  # MLE untouched
    assert top["count"] == 2 and top["total"] == 3
    lo, hi = top["ci95"]
    assert lo <= 2 / 3 <= hi


def test_context_entries_calibrated():
    model = build_context_model(list("ABCABC"), max_order=2)
    top = predict_with_context(model, ["A", "B"])[0]
    assert top["next"] == "C" and top["probability"] == 1.0
    assert top["count"] == 2 and top["total"] == 2
    assert top["order"] == 2


def _accuracy(predict_fn, train, test):
    hits = 0
    for i, actual in enumerate(test):
        hist = list(train) + list(test[:i])
        if predict_fn(hist) == actual:
            hits += 1
    return hits / len(test)


def test_context_beats_order1_on_second_order_data():
    # AABBAABB...: order-1 from A is a 50/50 coin; order-2 is exact.
    data = list("AABBAABBAABB")
    train, test = data[:8], data[8:]
    m = build_transition_matrix(train)
    cm = build_context_model(train, 2)
    acc1 = _accuracy(lambda h: (predict_next(h[-1], m) or [{"next": None}])[0]["next"],
                     train, test)
    acc2 = _accuracy(lambda h: (predict_with_context(cm, h[-2:]) or [{"next": None}])[0]["next"],
                     train, test)
    assert acc2 == 1.0
    assert acc2 > acc1


def test_abstain_on_random_data():
    rng = random.Random(11)
    data = [str(rng.randint(0, 14)) for _ in range(200)]
    out = Nexora().predict_next(data)
    assert out["source"] == "abstain" and out["next"] is None
    assert "abstain_threshold" in out["evidence"]


def test_abstain_on_thin_evidence():
    # Single observation of the deciding transition: honest engines wait.
    out = Nexora().predict_next(list("ABCD"))
    assert out["source"] == "abstain" and out["next"] is None
    assert "min_evidence" in out["evidence"]


def test_no_abstain_on_clean_cycle():
    out = Nexora().predict_next(list("ABCABCABC"))
    assert out["source"] in ("context", "markov")
    assert out["next"] == "A" and out["probability"] == 1.0


def test_abstain_configurable():
    rng = random.Random(11)
    data = [str(rng.randint(0, 14)) for _ in range(200)]
    out = Nexora(config={"abstain_threshold": 0.0, "min_evidence": 1}).predict_next(data)
    assert out["source"] in ("context", "markov")
    assert out["next"] is not None
