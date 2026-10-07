"""v3 batch C (I15-I19): analysis wrappers."""
import math

from nexora.api.engine import Nexora


def test_seasonality():
    nx = Nexora()
    s = nx.seasonality([float(i % 12) for i in range(120)])
    assert s["status"] == "FOUND" and s["period"] == 12
    assert nx.seasonality([1.0, 2.0])["status"] == "INSUFFICIENT_DATA"


def test_correlation():
    nx = Nexora()
    rows = [{"a": float(i), "b": float(2 * i)} for i in range(20)]
    c = nx.correlation(rows)
    assert c["status"] == "FOUND" and abs(c["pairs"][0]["r"]) > 0.99
    assert nx.correlation([1, 2, 3])["status"] == "INSUFFICIENT_DATA"


def test_regimes():
    nx = Nexora()
    r = nx.regimes([5.0] * 20 + [50.0] * 20)
    assert r["status"] == "FOUND" and r["count"] >= 1
    assert nx.regimes([1.0] * 5)["status"] == "INSUFFICIENT_DATA"


def test_change_points():
    nx = Nexora()
    c = nx.change_points([1.0] * 20 + [10.0] * 20)
    assert c["status"] == "FOUND" and c["count"] >= 1


def test_frequencies_transitions():
    nx = Nexora()
    f = nx.frequencies(list("AAABBC"))
    assert f["status"] == "FOUND"
    assert f["frequencies"][0] == {"value": "A", "count": 3, "fraction": 0.5}
    assert math.isclose(sum(x["fraction"] for x in f["frequencies"]), 1.0)
    t = nx.transitions(list("ABAB"))
    assert t["status"] == "FOUND"
    ab = [x for x in t["transitions"] if (x["from"], x["to"]) == ("A", "B")][0]
    assert ab["count"] == 2 and ab["probability"] == 1.0
    assert nx.transitions(["only"])["status"] == "NONE"
