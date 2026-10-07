"""v3 batch A (I5-I9): engine read-out helpers."""
from nexora.api.engine import Nexora


def test_describe():
    nx = Nexora()
    d = nx.describe([1.0, 2.0, 3.0])
    assert d["status"] == "FOUND" and d["n"] == 3
    assert nx.describe([])["status"] == "INSUFFICIENT_DATA"


def test_top_and_by_type():
    nx = Nexora()
    assert nx.top_patterns()["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    top = nx.top_patterns(2)
    assert top["status"] == "FOUND" and top["count"] <= 2
    confs = [p["confidence"] for p in top["patterns"]]
    assert confs == sorted(confs, reverse=True)
    assert nx.patterns_by_type("nope")["status"] == "NONE"


def test_stream_accessors():
    nx = Nexora()
    assert nx.stream_stats()["status"] == "NONE"
    assert nx.stream_changes()["count"] == 0
    nx.update([0.0] * 150 + [5.0] * 150)
    assert nx.stream_stats()["n"] == 300
    assert nx.stream_changes()["status"] == "FOUND"


def test_pattern_card():
    nx = Nexora()
    assert nx.pattern_card("P-999")["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    pid = nx.top_patterns(1)["patterns"][0]["id"]
    card = nx.pattern_card(pid)
    assert card["status"] == "FOUND" and pid in card["card"]


def test_solve():
    nx = Nexora()
    s = nx.solve([2, 4, 6, 8])
    assert s["status"] == "FOUND" and s["next"][0] == 10
    assert nx.solve([1])["status"] == "NONE"
