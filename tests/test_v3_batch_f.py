"""v3 batch F (I30-I34): numeric views."""
from nexora.api.engine import Nexora


def test_histogram():
    nx = Nexora()
    h = nx.histogram([1.0, 2.0, 3.0, 4.0], bins=2)
    assert h["status"] == "FOUND" and sum(b["count"] for b in h["bins"]) == 4
    assert nx.histogram([])["status"] == "INSUFFICIENT_DATA"


def test_zscores():
    nx = Nexora()
    z = nx.zscores([10.0] * 10 + [20.0])
    assert z["status"] == "FOUND" and len(z["zscores"]) == 11
    assert z["zscores"][-1]["z"] > 3.0
    assert nx.zscores([1.0])["status"] == "INSUFFICIENT_DATA"


def test_autocorr():
    nx = Nexora()
    a = nx.autocorr([float(i % 4) for i in range(40)], max_lag=8)
    assert a["status"] == "FOUND" and len(a["lags"]) == 8
    assert a["lags"][4] >= 0.89
    assert nx.autocorr([])["status"] == "INSUFFICIENT_DATA"


def test_moving_average():
    nx = Nexora()
    m = nx.moving_average([1.0, 2.0, 3.0, 4.0], window=2)
    assert m["status"] == "FOUND" and m["series"][-1] == 3.5
    assert nx.moving_average([])["status"] == "INSUFFICIENT_DATA"


def test_trend():
    nx = Nexora()
    t = nx.trend([2.0 * i for i in range(10)])
    assert t["status"] == "FOUND" and t["direction"] == "rising"
    assert abs(t["slope"] - 2.0) < 1e-9
    assert nx.trend([1.0])["status"] == "INSUFFICIENT_DATA"
