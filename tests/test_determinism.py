"""Workstream 10: determinism — same input, same output, every time.

Compares JSON-normalized results across two fresh engines (and two
processes would agree too: no randomness, no wall-clock, no dict
iteration anywhere in the result paths; shuffles use seeded local
RNGs only).
"""
import json

from nexora import Nexora


def _norm(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def test_discover_deterministic():
    data = list("ABCABCABCAXCABC")
    a, b = Nexora(), Nexora()
    r1, r2 = a.discover(list(data)), b.discover(list(data))
    assert _norm(r1["patterns"]) == _norm(r2["patterns"])
    assert _norm(r1["evidence"]) == _norm(r2["evidence"])
    assert r1["explanation"] == r2["explanation"]
    assert r1["status"] == r2["status"]


def test_anomalies_predict_deterministic():
    data = [round(10 + (i % 5) * 0.1 - 0.2, 2) for i in range(30)] + [25.0]
    a, b = Nexora(), Nexora()
    assert _norm(a.find_anomalies(list(data))) == _norm(b.find_anomalies(list(data)))
    seq = list("ABCABCABC")
    assert _norm(a.predict(list(seq))) == _norm(b.predict(list(seq)))
    assert _norm(a.predict_next(list(seq))) == _norm(b.predict_next(list(seq)))


def test_significance_seed_stable():
    data = list("ABC") * 12
    a, b = Nexora(), Nexora()
    p1 = [p.get("significance") for p in a.discover(data)["patterns"]]
    p2 = [p.get("significance") for p in b.discover(data)["patterns"]]
    assert p1 == p2
    c = Nexora(config={"sig_seed": 1234}).discover(data)["patterns"]
    p3 = [p.get("significance") for p in c]
    assert all((x is None) == (y is None) for x, y in zip(p1, p3))  # same shape


def test_streaming_deterministic():
    data = [float(i % 7) for i in range(300)] + [5.0] * 60
    a, b = Nexora(), Nexora()
    ra = [a.update(data[i:i + 50]) for i in range(0, len(data), 50)]
    rb = [b.update(data[i:i + 50]) for i in range(0, len(data), 50)]
    assert _norm(ra) == _norm(rb)
    assert _norm(a.stream_predict()) == _norm(b.stream_predict())


def test_match_ranking_deterministic():
    a, b = Nexora(), Nexora()
    a.discover(list("ABCABCABC"))
    b.discover(list("ABCABCABC"))
    assert _norm(a.match("Z")) == _norm(b.match("Z"))
    assert _norm(a.detect(list("ABC"))) == _norm(b.detect(list("ABC")))
