"""Loop 2 batch Q (I86-I90)."""
from nexora.api.engine import Nexora


def test_dunders():
    nx = Nexora()
    assert len(nx) == 0 and ("P-001" not in nx) and list(nx) == []
    nx.discover(list("ABCABCABC"))
    assert len(nx) > 0
    pid = nx.top_patterns(1)["patterns"][0]["id"]
    assert pid in nx and pid not in Nexora()
    assert "Nexora(patterns=" in repr(nx)


def test_api_catalog():
    nx = Nexora()
    a = nx.api()
    assert a["status"] == "FOUND"
    assert "discover" in a["methods"] and "forecast" in a["methods"]
    assert all(isinstance(v, str) for v in a["methods"].values())


def test_limits():
    nx = Nexora()
    lim = nx.limits()["limits"]
    assert lim["min_data"]["min_data_discover"] == 1
    assert lim["gates"]["match_threshold"] == 0.5
    assert lim["capabilities"]["robust"] is True


def test_anomaly_kinds():
    nx = Nexora()
    k = nx.anomaly_kinds([10.0] * 30 + [25.0])
    assert k["status"] == "FOUND" and sum(k["kinds"].values()) == k["count"]
