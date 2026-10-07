"""Loop 2 batch M (I66-I70)."""
import math

import pytest

from nexora.api.engine import Nexora


def test_motifs():
    nx = Nexora()
    m = nx.motifs(list("ABCABCABC"), size=3)
    assert m["status"] == "FOUND"
    assert m["motifs"][0]["motif"] == ["A", "B", "C"] and m["motifs"][0]["count"] == 3
    assert nx.motifs(["A"])["status"] == "INSUFFICIENT_DATA"


def test_rules():
    nx = Nexora()
    r = nx.rules(list("ABCABCABCABC"))
    assert r["status"] == "FOUND" and r["rules"]
    assert all(x["confidence"] >= 0.5 for x in r["rules"])
    with pytest.raises(ValueError):
        nx.rules(list("ABC"), min_confidence=2.0)


def test_centrality():
    nx = Nexora()
    c = nx.centrality(list("AABBAB"))
    assert c["status"] == "FOUND" and c["nodes"][0]["label"] == "A"


def test_entropy():
    nx = Nexora()
    e = nx.entropy(["A", "B"])
    assert e["status"] == "FOUND"
    assert e["bits"] == pytest.approx(1.0) and e["normalized"] == pytest.approx(1.0)
    assert nx.entropy(["A", "A"])["bits"] == pytest.approx(0.0)
    assert nx.entropy([])["status"] == "NONE"


def test_stationarity():
    nx = Nexora()
    assert nx.stationarity([5.0] * 20)["verdict"] == "stationary"
    assert nx.stationarity([1.0] * 10 + [100.0] * 10)["verdict"] == "drift"
    assert nx.stationarity([1.0, 2.0])["status"] == "INSUFFICIENT_DATA"
