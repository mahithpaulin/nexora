"""Loop 2 batch R (I91-I95): per-call overrides."""
import pytest

from nexora.api.engine import Nexora


def test_match_threshold():
    nx = Nexora()
    nx.discover(list("ABCABCABC"))
    assert nx.match("A", threshold=0.99)[0]["threshold"] == 0.99
    assert nx.match_top("A", threshold=0.0)["matched"] is True
    with pytest.raises(ValueError):
        nx.match("A", threshold=2.0)


def test_discover_max_n():
    nx = Nexora()
    d = nx.discover(list("ABCABCABC"), max_n=2)
    assert "ids" in d and d["status"] in ("FOUND", "NONE", "LOW_CONFIDENCE")
    with pytest.raises(ValueError):
        nx.discover(list("ABC"), max_n=11)


def test_robust_window_override():
    nx = Nexora()
    data = [10.0] * 30 + [25.0]
    assert nx.find_anomalies(data, robust_window=5)["count"] >= 1
    with pytest.raises(ValueError):
        nx.find_anomalies(data, robust_window=0)


def test_predict_order():
    nx = Nexora()
    p = nx.predict(list("ABCABCABC"), order=1)
    assert p["status"] == "FOUND"
    with pytest.raises(ValueError):
        nx.predict(list("ABCABC"), order=99)


def test_regimes_overrides():
    nx = Nexora()
    assert nx.regimes([5.0] * 20 + [50.0] * 20, size=10, k=2)["status"] == "FOUND"
    with pytest.raises(ValueError):
        nx.regimes([1.0] * 20, size=1)
