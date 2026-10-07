"""v3 batch D (I20-I24): predict perplexity, detect+anomalies, compare, summarize, forget."""
import math

from nexora.api.engine import Nexora


def test_perplexity():
    nx = Nexora()
    p = nx.predict(list("ABCABCABC"))
    assert p["evidence"]["perplexity"] is not None
    assert math.isclose(p["evidence"]["perplexity"], 2.0 ** p["log_loss"])


def test_detect_include_anomalies():
    nx = Nexora()
    base = nx.detect([10.0] * 30 + [25.0])
    assert "anomalies" not in base
    full = nx.detect([10.0] * 30 + [25.0], include_anomalies=True)
    assert full["anomaly_count"] >= 1
    assert full["anomaly_status"] == "FOUND"


def test_compare():
    a, b = Nexora(), Nexora()
    a.discover(list("ABCABCABC"))
    b.discover(list("XYZXYZXYZ"))
    d = a.compare(b)
    assert d["status"] == "FOUND"
    assert isinstance(d["added"], list) and isinstance(d["removed"], list)
    same = a.compare(a.export_patterns()["patterns"])
    assert same["removed"] == [] and same["added"] == []


def test_summarize():
    nx = Nexora()
    s = nx.summarize()
    assert s["patterns"] == 0 and s["stream_n"] == 0
    nx.discover(list("ABCABCABC"))
    nx.update([1.0])
    s = nx.summarize()
    assert s["patterns"] > 0 and s["stream_n"] == 1 and s["by_type"]


def test_forget():
    nx = Nexora()
    assert nx.forget("P-999")["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    pid = nx.top_patterns(1)["patterns"][0]["id"]
    assert nx.forget(pid)["forgotten"] is True
    assert nx.get_pattern(pid) is None
