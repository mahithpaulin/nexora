"""Loop 2 batch J (I51-I55)."""
from nexora.api.engine import Nexora


def test_explain_list():
    nx = Nexora()
    d = nx.discover(list("ABCABC"))
    text = nx.explain([d, d])
    assert text.startswith("1. ") and "\n2. " in text
    assert nx.explain([]) == "(no results)"


def test_report_patterns():
    nx = Nexora()
    assert nx.report_patterns()["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    r = nx.report_patterns(5)
    assert r["status"] == "FOUND" and r["markdown"]
    assert r["count"] == len(r["markdown"].split("---"))


def test_anomaly_report():
    nx = Nexora()
    assert nx.anomaly_report([1.0, 2.0])["status"] == "NONE"
    r = nx.anomaly_report([10.0] * 30 + [25.0])
    assert r["status"] == "FOUND" and "idx 30" in r["markdown"]


def test_coverage():
    nx = Nexora()
    assert nx.coverage([])["status"] == "INSUFFICIENT_DATA"
    nx.discover(list("ABCABCABC"))
    c = nx.coverage(list("ABCABCABC"))
    assert c["status"] == "FOUND" and 0.0 < c["fraction"] <= 1.0
    assert Nexora().coverage(["Q"] * 5)["fraction"] == 0.0


def test_sampling():
    nx = Nexora()
    rows = [{"value": 1.0, "t": i} for i in [0, 1, 2, 10, 11]]
    s = nx.sampling(rows)
    assert s["status"] == "FOUND" and s["median_dt"] == 1.0
    assert s["gaps"] == [{"after": 2, "dt": 8.0}]
    assert nx.sampling([{"value": 1.0, "t": 0}])["status"] == "INSUFFICIENT_DATA"
