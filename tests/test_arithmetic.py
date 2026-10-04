"""Closed-form sequence solving tests + engine wiring."""
import pytest

from nexora.discovery.arithmetic import (analyze_numeric_sequence,
                                         detect_additive_recurrence,
                                         detect_geometric,
                                         extrapolate_poly,
                                         finite_differences)
from nexora.api.engine import Nexora


def test_arithmetic_progression():
    r = analyze_numeric_sequence([2, 4, 6, 8])
    assert r["kind"] == "arithmetic" and r["next"] == [10]
    assert r["confidence"] == 1.0


def test_quadratic():
    r = analyze_numeric_sequence([1, 4, 9, 16])
    assert r["kind"] == "quadratic" and r["next"] == [25]


def test_geometric():
    r = analyze_numeric_sequence([3, 6, 12, 24])
    assert r["kind"] == "geometric" and r["next"] == [48]
    assert detect_geometric([2, 3, 4]) is None


def test_fibonacci_like():
    r = analyze_numeric_sequence([1, 1, 2, 3, 5, 8])
    assert r["kind"] == "fibonacci_like" and r["next"] == [13]
    assert detect_additive_recurrence([1, 2, 4]) is None


def test_unknown_and_edges():
    r = analyze_numeric_sequence([2, 7, 1, 9, 4])
    assert r["kind"] == "unknown" and r["next"] == [] and r["confidence"] == 0.0
    r = analyze_numeric_sequence([2, 7, 1, 9])  # degenerate cubic rejected
    assert r["kind"] == "unknown"
    r = analyze_numeric_sequence([1, 2])
    assert r["kind"] == "unknown"
    with pytest.raises(ValueError):
        analyze_numeric_sequence(["a", "b", "c"])
    fd = finite_differences([2, 4, 6, 8])
    assert fd["constant_order"] == 1
    assert extrapolate_poly([2, 4, 6, 8], 1, 3) == [10, 12, 14]


def test_engine_predict_extrapolates():
    pr = Nexora().predict([2, 4, 6, 8])
    assert pr["extrapolation"]["kind"] == "arithmetic"
    assert pr["extrapolation"]["next"][0] == 10


def test_engine_discovers_arithmetic_pattern():
    disc = Nexora().discover([1, 1, 2, 3, 5, 8])
    arith = [p for p in disc["patterns"] if p["type"] == "arithmetic"]
    assert arith and arith[0]["features"]["kind"] == "fibonacci_like"
    assert arith[0]["features"]["next"] == 13
