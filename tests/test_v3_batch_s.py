"""Loop 2 final batch S (I96-I99; I100 is the release)."""
import json

from nexora.api.engine import Nexora


def test_periodogram():
    nx = Nexora()
    p = nx.periodogram([float(i % 4) for i in range(40)])
    assert p["status"] == "FOUND" and p["best_lag"] == 4


def test_bars():
    nx = Nexora()
    b = nx.bars([1.0, 2.0, 3.0, 4.0])
    assert b["status"] == "FOUND" and len(b["spark"]) == 4
    assert b["spark"][0] != b["spark"][-1]
    assert nx.bars([])["status"] == "INSUFFICIENT_DATA"


def test_records():
    nx = Nexora()
    r = nx.records(["A", 1.0, None])
    assert r["status"] == "FOUND" and r["count"] == 3
    json.dumps(r["records"])
    assert nx.records([])["status"] == "NONE"


def test_jaccard():
    nx = Nexora()
    assert nx.jaccard(list("ABC"), list("ABC"))["jaccard"] == 1.0
    assert nx.jaccard(list("ABC"), list("XYZ"))["jaccard"] == 0.0
    assert nx.jaccard([], [])["status"] == "INSUFFICIENT_DATA"
