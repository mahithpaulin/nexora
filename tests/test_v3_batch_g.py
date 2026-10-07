"""v3 batch G (I35-I39): explain_pattern, state_signature, per-call overrides."""
import pytest

from nexora.api.engine import Nexora


def test_explain_pattern():
    nx = Nexora()
    assert nx.explain_pattern("P-999")["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    pid = nx.top_patterns(1)["patterns"][0]["id"]
    nar = nx.explain_pattern(pid)
    assert nar["status"] == "FOUND"
    assert pid in nar["narrative"] and "confidence" in nar["narrative"]


def test_state_signature():
    a, b = Nexora(), Nexora()
    assert a.state_signature()["signature"] == b.state_signature()["signature"]
    a.discover(list("ABCABCABC"))
    assert a.state_signature()["signature"] != b.state_signature()["signature"]
    c = Nexora()
    c.discover(list("ABCABCABC"))
    assert a.state_signature()["signature"] == c.state_signature()["signature"]


def test_discover_min_support_override():
    nx = Nexora()
    assert nx.discover(["A"] * 4, min_support=5)["count"] == 0
    assert nx.discover(["A"] * 4, min_support=2)["count"] >= 1
    with pytest.raises(ValueError):
        nx.discover(["A"], min_support=0)


def test_anomaly_z_override():
    nx = Nexora()
    data = [10.0] * 30 + [25.0]
    # z=100 silences the classical detector; the robust median/MAD
    # path (own threshold) may still fire — but never as statistical.
    hi = nx.find_anomalies(data, z_threshold=100.0)
    assert all("statistical" not in a["kind"] for a in hi["anomalies"])
    assert nx.find_anomalies(data, z_threshold=1.0)["count"] >= 1
    with pytest.raises(ValueError):
        nx.find_anomalies(data, z_threshold=-2.0)


def test_predict_top_k():
    nx = Nexora()
    p = nx.predict(list("ABCABCABCABC"), top_k=1)
    assert len(p["predictions"]) <= 1 and len(p["context"]) <= 1
    with pytest.raises(ValueError):
        nx.predict(list("ABCABC"), top_k=0)
