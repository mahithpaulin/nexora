"""Loop 2 batch P (I81-I85)."""
import pytest

from nexora.api.engine import Nexora


def test_simulate():
    nx = Nexora()
    w1 = nx.simulate(list("ABCABCABC"), steps=6, seed=7)
    assert w1["status"] == "FOUND" and len(w1["walk"]) == 6
    assert set(w1["walk"]) <= {"A", "B", "C"}
    assert nx.simulate(list("ABCABCABC"), steps=6, seed=7) == w1
    assert nx.simulate(["A"])["status"] == "INSUFFICIENT_DATA"
    with pytest.raises(ValueError):
        nx.simulate(list("ABC"), steps=0)


def test_sequence_prob():
    nx = Nexora()
    p = nx.sequence_prob(list("ABABAB"), ["A", "B", "A"])
    assert p["status"] == "FOUND" and p["probability"] == 1.0
    z = nx.sequence_prob(list("ABABAB"), ["A", "C"])
    assert z["probability"] == 0.0
    assert nx.sequence_prob([], ["A", "B"])["status"] == "INSUFFICIENT_DATA"


def test_predict_proba():
    nx = Nexora()
    p = nx.predict_proba(list("ABCABCABC"), "A", current="C")
    assert p["status"] == "FOUND" and p["probability"] == 1.0
    assert nx.predict_proba(list("ABCABCABC"), "ZZZ")["status"] == "NONE"


def test_divergence():
    nx = Nexora()
    assert nx.divergence(list("AAA"), list("AAA"))["l1"] == 0.0
    d = nx.divergence(list("AAA"), list("BBB"))
    assert d["l1"] == 2.0
    assert nx.divergence([], [1])["status"] == "INSUFFICIENT_DATA"


def test_seasonal_forecast():
    nx = Nexora()
    assert nx.seasonal_forecast([1.0, 2.0])["status"] == "NONE"
    nx.discover([float(i % 4) for i in range(80)])
    f = nx.seasonal_forecast([float(i % 4) for i in range(80)], steps=2)
    assert f["status"] == "FOUND" and len(f["steps"]) == 2
