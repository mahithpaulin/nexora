"""v3 final batch I (I45-I49): dtw, align, reduce, discretize (+I48 demo, I50 release)."""
import pytest

from nexora.api.engine import Nexora


def test_dtw():
    nx = Nexora()
    d = nx.dtw([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert d["status"] == "FOUND" and d["distance"] == 0.0
    assert d["similarity"] == 1.0
    assert nx.dtw([], [1.0])["status"] == "INSUFFICIENT_DATA"


def test_align():
    nx = Nexora()
    a = [float((i * 7) % 11) for i in range(40)]
    b = [0.0, 0.0] + a[:-2]  # b trails a by 2
    r = nx.align(a, b, max_lag=5)
    assert r["status"] == "FOUND" and r["lag"] == 2 and r["r"] > 0.9
    assert nx.align([1.0], [2.0])["status"] == "INSUFFICIENT_DATA"


def test_reduce():
    nx = Nexora()
    r = nx.reduce([float(i % 10) for i in range(64)], size=8)
    assert r["status"] == "FOUND" and r["components"]
    assert nx.reduce([1.0, 2.0])["status"] == "INSUFFICIENT_DATA"
    with pytest.raises(ValueError):
        nx.reduce([1.0] * 40, size=1)


def test_discretize():
    nx = Nexora()
    d = nx.discretize([1.0, 2.0, 3.0, 4.0], bins=2)
    assert d["status"] == "FOUND" and len(d["labels"]) == 4
    assert d["labels"][0] != d["labels"][-1]
    assert nx.discretize([])["status"] == "INSUFFICIENT_DATA"
    assert nx.discretize([5.0] * 3)["bins"] == 1
