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
