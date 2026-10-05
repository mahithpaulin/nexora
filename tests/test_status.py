"""Workstream 5: honest result statuses.

Every dict-result carries status in {FOUND, NONE, INSUFFICIENT_DATA,
LOW_CONFIDENCE} plus a status_reason citing numbers. Minimum-data
thresholds live in config (fail-fast). Nothing is fabricated from
too little data. match() is the documented exception: it returns a
ranked list whose per-entry `matched` verdict is the status.
"""
import pytest

from nexora import Nexora
from nexora.core.errors import InvalidConfigError


def test_discover_statuses():
    assert Nexora().discover([])["status"] == "INSUFFICIENT_DATA"
    assert "need >= 1" in Nexora().discover([])["status_reason"]
    none = Nexora().discover(["only"])
    assert none["status"] == "NONE"
    assert "min_support=3" in none["status_reason"]
    found = Nexora().discover(list("ABC") * 10)
    assert found["status"] == "FOUND"
    assert "10 pattern" not in found["status_reason"]  # stored, not repo total
    low = Nexora().discover(list("ABCABCABC"))
    assert low["status"] == "LOW_CONFIDENCE"
    assert "sig_min_n" in low["status_reason"]


def test_anomaly_statuses():
    nx = Nexora()
    assert nx.find_anomalies([])["status"] == "INSUFFICIENT_DATA"
    assert nx.find_anomalies([1.0])["status"] == "INSUFFICIENT_DATA"
    assert "need >= 2" in nx.find_anomalies([1.0])["status_reason"]
    assert nx.find_anomalies([7.0] * 40)["status"] == "NONE"
    assert nx.find_anomalies([10.0] * 30 + [25.0])["status"] == "FOUND"


def test_predict_statuses():
    nx = Nexora()
    assert nx.predict([])["status"] == "INSUFFICIENT_DATA"
    assert nx.predict(["only"])["status"] == "INSUFFICIENT_DATA"
    # One observation of the deciding transition: reported, not trusted.
    thin = nx.predict(["A", "B", "A"])
    assert thin["status"] == "LOW_CONFIDENCE"
    assert "N=1" in thin["status_reason"]
    assert nx.predict(["A", "B", "A", "B", "A"])["status"] == "FOUND"
    assert nx.predict(list("ABCABCABC"))["status"] == "FOUND"
    import random
    rng = random.Random(11)
    rnd = nx.predict([str(rng.randint(0, 14)) for _ in range(200)])
    assert rnd["status"] == "LOW_CONFIDENCE"
    assert "abstain_threshold" in rnd["status_reason"]
    # Unknown current: Markov has nothing, context backoff answers from
    # the tail with a weak top guess -> LOW_CONFIDENCE (honest).
    zzz = nx.predict(["A", "B", "C"], current="ZZZ")
    assert zzz["status"] == "LOW_CONFIDENCE"


def test_predict_next_status_mirrors_source():
    nx = Nexora()
    assert nx.predict_next([])["status"] == "NONE"
    assert nx.predict_next(list("ABCABCABC"))["status"] == "FOUND"
    import random
    rng = random.Random(11)
    out = nx.predict_next([str(rng.randint(0, 14)) for _ in range(200)])
    assert out["status"] == "LOW_CONFIDENCE" and out["source"] == "abstain"
    assert nx.predict_next([2, 4, 6, 8])["status"] == "FOUND"


def test_detect_get_history_quality_load_status():
    nx = Nexora()
    assert nx.detect([])["status"] == "INSUFFICIENT_DATA"
    assert nx.detect(list("ABC") * 10)["status"] == "FOUND"
    assert nx.get_history("P-999")["status"] == "NONE"
    nx.discover(list("ABCABCABC"))
    assert nx.get_history("P-001")["status"] == "FOUND"
    assert nx.quality([])["status"] == "INSUFFICIENT_DATA"
    assert nx.quality([1, 2, 3])["status"] == "FOUND"


def test_stream_statuses():
    nx = Nexora()
    assert nx.update([])["status"] == "NONE"
    assert nx.update([1.0, 2.0])["status"] == "FOUND"
    assert nx.stream_predict()["status"] == "NONE"
    nx.update(list("ABAB"))
    assert nx.stream_predict()["status"] == "FOUND"


def test_match_keeps_list_shape_with_per_entry_verdict():
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABCABC"))
    res = nx.match("Z")
    assert isinstance(res, list)
    assert all("matched" in m and "status" not in m for m in res)


def test_min_data_thresholds_validated():
    for bad in ({"min_data_discover": 0}, {"min_data_anomalies": -1},
                {"min_data_predict": "x"}):
        try:
            Nexora(config=bad)
        except InvalidConfigError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected InvalidConfigError for %r" % bad)
    nx = Nexora(config={"min_data_anomalies": 10})
    assert nx.find_anomalies([1.0] * 9)["status"] == "INSUFFICIENT_DATA"
    assert "need >= 10" in nx.find_anomalies([1.0] * 9)["status_reason"]
