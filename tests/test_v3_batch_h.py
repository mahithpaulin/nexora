"""v3 batch H (I40-I44): stream top_k, config_help, discover ids, grade, graph."""
import pytest

from nexora.api.engine import Nexora


def test_stream_top_k():
    nx = Nexora()
    nx.update(list("ABABAB"))
    assert len(nx.stream_predict(top_k=1)["predictions"]) <= 1
    with pytest.raises(ValueError):
        nx.stream_predict(top_k=0)


def test_config_help():
    nx = Nexora()
    all_docs = nx.config_help()
    assert all_docs["status"] == "FOUND" and "z_threshold" in all_docs["docs"]
    one = nx.config_help("z_threshold")
    assert one["status"] == "FOUND" and one["doc"]
    assert nx.config_help("nope")["status"] == "NONE"


def test_discover_ids():
    nx = Nexora()
    d = nx.discover(list("ABCABCABC"))
    assert d["ids"] == [p["id"] for p in d["patterns"]]
    assert nx.discover([])["ids"] == []


def test_quality_grade():
    nx = Nexora()
    assert nx.quality([1, 2, 3, 4, 5])["grade"] == "A"
    assert nx.quality([7, 7, 7])["grade"] in ("B", "C")


def test_graph():
    nx = Nexora()
    g = nx.graph(list("AABBAB"))
    assert g["status"] == "FOUND" and g["nodes"] >= 2
    assert g["top_edges"]
