"""Loop 2 batch K (I56-I60)."""
from nexora.api.engine import Nexora


def test_forecast():
    nx = Nexora()
    f = nx.forecast([2, 4, 6, 8], steps=2)
    assert f["status"] == "FOUND" and f["steps"] == [10, 12]
    assert f["source"].startswith("arithmetic")
    t = nx.forecast([1.0, 3.0, 2.0, 4.0], steps=1)
    assert t["status"] == "FOUND" and len(t["steps"]) == 1
    assert nx.forecast([1.0])["status"] == "INSUFFICIENT_DATA"


def test_backtest():
    nx = Nexora()
    b = nx.backtest(list("ABCABCABC"))
    assert b["status"] == "FOUND" and b["accuracy"] == 5 / 8  # cold-start misses
    assert nx.backtest(list("ABC") * 10)["accuracy"] >= 0.85
    assert nx.backtest(["A"])["status"] == "INSUFFICIENT_DATA"


def test_surprises():
    nx = Nexora()
    s = nx.surprises(list("AAAAABAAAA"))
    assert s["status"] == "FOUND"
    assert s["points"][0]["label"] == "B"
    assert nx.surprises(["A"])["status"] == "INSUFFICIENT_DATA"


def test_markov_table():
    nx = Nexora()
    m = nx.markov_table(list("ABAB"))
    assert m["status"] == "FOUND"
    assert m["table"]["A"]["B"] == 1.0
    assert nx.markov_table([])["status"] == "INSUFFICIENT_DATA"


def test_vocabulary():
    nx = Nexora()
    v = nx.vocabulary(list("AAABC"))
    assert v["status"] == "FOUND" and v["distinct"] == 3
    assert v["labels"][0] == {"label": "A", "count": 3, "fraction": 0.6}
    assert nx.vocabulary([])["status"] == "NONE"
