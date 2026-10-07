"""v3 batch B (I10-I14): friendlier inputs + engine management."""
from nexora.api.engine import Nexora


def test_csv_string_routes_via_parser():
    nx1, nx2 = Nexora(), Nexora()
    assert nx1.discover("A,B,C,A,B,C,A,B,C")["count"] == \
        nx2.discover(["A", "B", "C"] * 3)["count"]
    st = nx1.find_anomalies("1,2,3")["evidence"]["stats"]
    assert st.get("n", st.get("count")) == 3


def test_plain_string_stays_single():
    nx = Nexora()
    assert nx.describe("ABC")["n"] == 1
    assert nx.describe("ABC")["status"] == "FOUND"


def test_configure_and_reset():
    import pytest
    nx = Nexora()
    out = nx.configure(z_threshold=5.0, typo_key=1)
    assert out["applied"] == ["z_threshold"] and out["unknown"] == ["typo_key"]
    assert nx.config["z_threshold"] == 5.0
    assert nx.configure()["applied"] == []
    with pytest.raises(Exception):
        nx.configure(z_threshold=-1.0)
    nx.discover(list("ABCABCABC"))
    nx.update([1.0, 2.0])
    assert nx.reset()["status"] == "FOUND"
    assert nx.top_patterns()["status"] == "NONE"
    assert nx.stream_stats()["status"] == "NONE"


def test_export_import_roundtrip():
    import json
    a = Nexora()
    assert a.export_patterns()["status"] == "NONE"
    a.discover(list("ABCABCABC"))
    exp = a.export_patterns()
    assert exp["status"] == "FOUND"
    json.dumps(exp["patterns"])  # serializable
    b = Nexora()
    imp = b.import_patterns(exp["patterns"])
    assert imp["status"] == "FOUND" and imp["imported"] == exp["count"]
    assert b.top_patterns()["count"] == a.top_patterns()["count"]
    assert b.import_patterns([None, 42])["status"] == "NONE"


def test_min_severity_filter():
    import pytest
    nx = Nexora()
    data = [10.0] * 30 + [25.0]
    all_hits = nx.find_anomalies(data)["anomalies"]
    assert all_hits, "baseline spike must flag"
    crit = nx.find_anomalies(data, min_severity="critical")
    assert len(crit["anomalies"]) <= len(all_hits)
    assert all(a["severity"] == "critical" for a in crit["anomalies"])
    assert nx.find_anomalies(data, min_severity="low")["count"] == len(all_hits)
    with pytest.raises(ValueError, match="low/medium/high/critical"):
        nx.find_anomalies(data, min_severity="extreme")


def test_match_top():
    nx = Nexora()
    assert nx.match_top("A")["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    hit = nx.match_top("A")
    assert hit["status"] == "FOUND" and hit["matched"] is True
    assert hit["ranked"] >= 1
    miss = nx.match_top("ZZZ-nope")
    assert miss["status"] == "NONE" and miss["matched"] is False
