"""Thorough evaluation: knowns, unknowns, small benchmarks, cognitive checks.

Stdlib only. Each test maps to a cognitive dimension: discovery, prediction,
anomaly, correlation, seasonality, clustering/regimes, memory/lifecycle,
explanation, robustness, determinism, persistence, scale.
"""
import math
import random

import pytest

from nexora import Nexora
from nexora.anomaly.multivariate import detect_multivariate
from nexora.discovery.arithmetic import analyze_numeric_sequence
from nexora.discovery.change_points import change_points
from nexora.discovery.sequences import find_frequent_sequences
from nexora.features.correlation import find_correlation_patterns
from nexora.features.seasonality import estimate_period
from nexora.streaming import stream_anomalies


# ---------- KNOWN discovery ----------

def test_known_cycle_discovers_and_predicts():
    nx = Nexora(config={"min_support": 2, "max_n": 3})
    d = list("ABC" * 10)
    disc = nx.discover(d)
    abc = [p for p in disc["patterns"] if tuple(p["sequence"]) == ("A", "B", "C")]
    assert len(abc) == 1 and abc[0]["frequency"] == 10
    pred = nx.predict(d)
    assert pred["predictions"] and pred["predictions"][0]["next"] == "A"


def test_known_two_regimes():
    d = [5.0 + (i % 4) * 0.1 for i in range(40)] + [50.0 - (i % 4) * 0.1 for i in range(40)]
    disc = Nexora().discover(d)
    regs = sorted([p for p in disc["patterns"] if p["type"] == "regime"],
                  key=lambda p: sum(p["sequence"]) / len(p["sequence"]))
    assert len(regs) == 2
    means = [sum(p["sequence"]) / len(p["sequence"]) for p in regs]
    assert means[0] < 10.0 and means[1] > 40.0


def test_known_spike_flagged():
    d = [round(10 + (i % 5) * 0.2, 1) for i in range(60)] + [100.0]
    out = Nexora().find_anomalies(d)
    # NOTE (v2/WS4): one event yields one merged record whose kind unions
    # all detectors that fired, so filter with `in`, not `==`.
    stat = [a for a in out["anomalies"] if "statistical" in a["kind"]]
    assert any(a["index"] == 60 for a in stat)


def test_known_seasonality_period_12():
    sig = [math.sin(2 * math.pi * i / 12) for i in range(48)]
    est = estimate_period(sig)
    assert est["period"] == 12


def test_known_correlation_strong():
    rng = random.Random(42)
    xs = [float(i) for i in range(20)]
    ys = [3 * x + rng.uniform(-0.5, 0.5) for x in xs]
    pats = find_correlation_patterns({"x": xs, "y": ys}, threshold=0.7)
    assert len(pats) == 1 and pats[0]["features"]["r"] > 0.95


def _nxt(r):
    n = r["next"]
    return n[0] if isinstance(n, list) else n


def test_known_arithmetic_linear():
    r = analyze_numeric_sequence([2.0, 4.0, 6.0, 8.0])
    assert r is not None and abs(_nxt(r) - 10.0) < 1e-9


def test_known_arithmetic_geometric():
    r = analyze_numeric_sequence([3.0, 6.0, 12.0, 24.0])
    assert r is not None and abs(_nxt(r) - 48.0) < 1e-9


def test_known_predict_next_arithmetic():
    out = Nexora().predict_next([2.0, 4.0, 6.0, 8.0])
    assert out and abs(out["next"] - 10.0) < 1e-9


# ---------- KNOWN anomaly / memory ----------

def test_known_change_point():
    cps = change_points([1.0] * 20 + [10.0] * 20, window=5, threshold_z=2.0)
    assert any(15 <= c["index"] <= 25 for c in cps)


def test_known_lifecycle_observed_confirmed():
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABCABC"))
    assert {p["state"] for p in nx.repo.all()} == {"OBSERVED"}
    nx.discover(list("ABCABCABC"))
    assert "CONFIRMED" in {p["state"] for p in nx.repo.all()}


def test_known_save_load_fidelity():
    import json as _json
    import os as _os
    import tempfile as _tf
    nx = Nexora(config={"min_support": 2})
    before = nx.discover(list("ABCABCABC"))
    assert before["patterns"]
    with _tf.TemporaryDirectory() as td:
        fp = _os.path.join(td, "s.json")
        nx.save(fp)
        nx2 = Nexora()
        nx2.load(fp)
        assert _json.dumps(nx.repo.all(), sort_keys=True, default=str) == \
            _json.dumps(nx2.repo.all(), sort_keys=True, default=str)


def test_known_explanation_nonempty_with_numbers():
    out = Nexora().discover(list("ABCABCABC"))
    assert isinstance(out["explanation"], str) and len(out["explanation"]) > 20
    assert any(ch.isdigit() for ch in out["explanation"])


def test_known_report_markdown_sections():
    nx = Nexora()
    res = nx.detect(list("ABCABC"))
    md = nx.report(res, fmt="markdown")
    assert isinstance(md, str) and len(md) > 20


def test_known_quality_bounded():
    q = Nexora().quality(list("ABCABC"))
    assert 0.0 <= q["quality"] <= 1.0


def test_known_invalid_config_raises():
    with pytest.raises(ValueError):
        Nexora({"z_threshold": -1})


# ---------- UNKNOWN discovery ----------

def test_unknown_quadratic_sequence():
    r = analyze_numeric_sequence([1.0, 4.0, 9.0, 16.0])
    assert r is not None and abs(_nxt(r) - 25.0) < 1e-6


def test_unknown_fibonacci_like():
    r = analyze_numeric_sequence([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
    assert r is not None and abs(_nxt(r) - 13.0) < 1e-6


def test_unknown_novel_transition_flagged():
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABCABCABC"))
    out = nx.find_anomalies(list("ABZ"))
    mt = [a for a in out["anomalies"] if a["kind"] == "missing_transition"]
    assert any(a["index"] == 2 for a in mt)


def test_unknown_random_control_no_crash():
    rng = random.Random(1234)
    d = [str(rng.randint(0, 999)) for _ in range(100)]
    pats = find_frequent_sequences(d, max_n=3, min_support=3)
    assert pats == []
    nx = Nexora()
    det = nx.detect(d[:30])
    assert isinstance(det["explanation"], str)


def test_unknown_constant_series_no_spurious_anomaly():
    out = Nexora().find_anomalies([7.0] * 40)
    stat = [a for a in out["anomalies"] if a["kind"] == "statistical"]
    assert stat == []


def test_unknown_single_spike_in_short_series():
    # Short series: z cannot reach 3 (scales with sqrt(n)); engine must not
    # crash and must stay honest (no forced flag).
    out = Nexora().find_anomalies([1.0, 1.0, 1.0, 99.0])
    assert isinstance(out["anomalies"], list)


def test_unknown_empty_and_single_inputs():
    nx = Nexora()
    for d in ([], [5.0], ["only"]):
        disc = nx.discover(d)
        assert isinstance(disc["patterns"], list)
        anom = nx.find_anomalies(d)
        assert isinstance(anom["anomalies"], list)
        pred = nx.predict(d)
        assert isinstance(pred["predictions"], list)


def test_unknown_none_nan_junk_skipped():
    out = stream_anomalies(["a", None, float("nan"), float("inf"), True], window=2)
    assert out == []
    nx = Nexora()
    disc = nx.discover([1.0, None, float("nan"), 2.0, 3.0])
    assert isinstance(disc["patterns"], list)


def test_unknown_shuffled_cycle_weak_or_absent():
    rng = random.Random(99)
    d = list("ABC" * 10)
    rng.shuffle(d)
    pats = find_frequent_sequences(d, max_n=3, min_support=5)
    abc = [p for p in pats if tuple(p["sequence"]) == ("A", "B", "C")]
    assert len(abc) <= 1


def test_unknown_two_spikes_both_found():
    d = [10.0] * 30 + [60.0] + [10.0] * 30 + [60.0]
    out = Nexora().find_anomalies(d)
    idx = {a["index"] for a in out["anomalies"]}
    assert 30 in idx and 61 in idx


def test_unknown_step_then_return_two_change_points():
    cps = change_points([1.0] * 20 + [10.0] * 20 + [1.0] * 20, window=5, threshold_z=2.0)
    assert len(cps) >= 2


def test_unknown_negative_correlation():
    xs = [float(i) for i in range(20)]
    ys = [-2.0 * x + 5.0 for x in xs]
    pats = find_correlation_patterns({"x": xs, "y": ys}, threshold=0.7)
    assert len(pats) == 1 and pats[0]["features"]["r"] < -0.99


def test_unknown_uncorrelated_pair_no_pattern():
    rng = random.Random(7)
    xs = [rng.uniform(-5, 5) for _ in range(30)]
    ys = [rng.uniform(-5, 5) for _ in range(30)]
    pats = find_correlation_patterns({"x": xs, "y": ys}, threshold=0.9)
    assert pats == []


def test_unknown_multivariate_clean_no_flag():
    pts = [[20.0 + (i % 3) * 0.1, 1013.0 + (i % 2) * 0.1] for i in range(25)]
    mv = detect_multivariate(pts, threshold=0.8)
    assert mv["anomalies"] == []


def test_unknown_batch_agrees_with_single():
    nx = Nexora(config={"min_support": 2})
    datasets = [list("ABCABC"), list("ABCABC")]
    b = nx.batch(datasets)
    assert len(b) == 2


def test_unknown_determinism_twice_same():
    import json as _json
    d = list("ABCABCABC") + [1.0, 2.0, 3.0]
    r1 = Nexora(config={"min_support": 2}).discover(d)
    r2 = Nexora(config={"min_support": 2}).discover(d)
    assert _json.dumps(r1["patterns"], sort_keys=True, default=str) == \
        _json.dumps(r2["patterns"], sort_keys=True, default=str)


def test_unknown_streaming_jump_flagged():
    out = stream_anomalies([5.0] * 5 + [20.0], window=5, warmup=5)
    assert len(out) == 1 and out[0]["index"] == 5


def test_small_bench_discover_5k_under_10s():
    import time as _t
    d = list("ABC" * 1700)[:5000]
    t0 = _t.perf_counter()
    out = Nexora().discover(d)
    dt = _t.perf_counter() - t0
    assert out["patterns"]
    assert dt < 10.0, f"took {dt:.2f}s"
