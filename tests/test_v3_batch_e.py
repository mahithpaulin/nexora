"""v3 batch E (I25-I29): prune, dict-batch, strict fmt, missing_runs, runs."""
import pytest

from nexora.api.engine import Nexora


def test_prune():
    nx = Nexora()
    assert nx.prune()["remaining"] == 0
    nx.discover(list("ABCABCABC"))
    left = nx.prune(max_total=1000)["remaining"]
    assert left == nx.top_patterns(1000)["count"]
    with pytest.raises(ValueError):
        nx.prune(max_total="many")


def test_dict_batch():
    nx = Nexora()
    out = nx.batch({"first": ["A", "B", "C"], "second": [1.0, 2.0]})
    assert {o["index"] for o in out} == {"first", "second"}
    assert all(o["error"] is None for o in out)


def test_report_strict_fmt():
    nx = Nexora()
    res = nx.detect(list("ABCABC"))
    assert isinstance(nx.report(res, fmt="bogus") if False else nx.report(res), str)
    with pytest.raises(ValueError, match="markdown.*text"):
        nx.report(res, fmt="bogus")


def test_missing_runs():
    nx = Nexora()
    m = nx.missing_runs([1.0, None, None, 2.0, None])
    assert m["status"] == "FOUND"
    assert m["runs"] == [{"start": 1, "end": 2, "length": 2},
                         {"start": 4, "end": 4, "length": 1}]
    assert nx.missing_runs([1.0, 2.0])["status"] == "NONE"


def test_runs():
    nx = Nexora()
    r = nx.runs(["A", "A", "B", "B", "B", "A"])
    assert r["count"] == 3
    assert r["runs"][1] == {"value": "B", "start": 2, "end": 4, "count": 3}
    assert nx.runs([])["status"] == "NONE"
