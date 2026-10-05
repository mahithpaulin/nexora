"""Nexora v2 tests: streaming hardening, detector/config/repository/
clustering/report upgrades, arithmetic reorder, engine wiring
(score_detail, new_patterns, skipped evidence, predict_next, trend
forecast, max_period, multivariate end-index). Stdlib only.
"""
import hashlib
import json
import math

import pytest

from nexora.streaming import (RunningStats, SlidingStats, stream_anomalies,
                              streaming_describe)
from nexora.anomaly.detector import detect
from nexora.memory.repository import PatternRepository, signature_of
from nexora.discovery.clustering import (agglomerative, dbscan,
                                         find_regimes, kmeans)
from nexora.discovery.arithmetic import analyze_numeric_sequence
from nexora.explanation.report import render_markdown, render_text
from nexora.anomaly.multivariate import detect_multivariate
from nexora.features.temporal import detect_trend
from nexora.features.seasonality import estimate_period
from nexora.ingestion.quality import quality_report
from nexora.prediction.markov import build_transition_matrix, predict_next
from nexora.prediction.context import (build_context_model,
                                       predict_with_context)
from nexora.api.engine import Nexora


def _blobs():
    return [[0.0], [0.1], [-0.1], [10.0], [10.1], [9.9]]


# ---------- streaming hardening ----------

def test_streaming_skips_str_bool_inf():
    assert stream_anomalies(["a", "b", True, float("inf")], window=2) == []
    assert stream_anomalies(["a", "b", True, float("inf")], window=2, warmup=0) == []
    d = streaming_describe(["a", "b", True, float("inf")], window=2)
    assert d["n"] == 0 and d["missing"] == 4
    rs = RunningStats()
    for x in ["a", True, float("inf"), float("-inf"), None, float("nan")]:
        rs.update(x)
    assert rs.n == 0 and rs.missing == 6
    sl = SlidingStats(3)
    for x in ["a", True, float("inf")]:
        sl.update(x)
    assert sl.n == 0 and sl.missing == 3 and sl.mean is None


def test_streaming_zero_variance_jump_flagged():
    out = stream_anomalies([5.0] * 5 + [20.0], window=5, warmup=5)
    assert len(out) == 1 and out[0]["index"] == 5
    assert out[0]["z"] == float("inf") and out[0]["score"] == 1.0
    assert "zero-variance baseline jump" in out[0]["explanation"]
    assert stream_anomalies([5.0] * 6, window=5, warmup=5) == []


# ---------- detector / repository / clustering ----------

def test_detector_bad_threshold_raises():
    rows = [{"index": 0, "value": 0.0, "label": "0"}]
    for bad in (0, 0.0, -1, -2.5):
        with pytest.raises(ValueError):
            detect(rows, {"mean": 0.0, "stdev": 1.0}, z_threshold=bad)


def test_repository_deepcopy_isolation():
    repo = PatternRepository()
    pid = repo.add({"type": "t", "features": {"v": [1, 2]},
                    "sequence": [1], "frequency": 1,
                    "occurrences": [0], "confidence": 0.5})
    g = repo.get(pid)
    g["features"]["v"].append(99)
    g["occurrences"].append(99)
    assert repo.get(pid)["features"] == {"v": [1, 2]}
    a = repo.all()
    a[0]["features"]["v"].append(100)
    assert repo.get(pid)["features"] == {"v": [1, 2]}


def test_repository_sha256_merge_still_works():
    p = {"type": "sequential", "features": {"ngram": ["A", "B"]},
         "sequence": ["A", "B"], "frequency": 3,
         "occurrences": [0], "confidence": 0.5}
    sig = signature_of(p)
    assert len(sig) == 12 and all(c in "0123456789abcdef" for c in sig)
    payload = json.dumps({"type": "sequential",
                          "features": {"ngram": ["A", "B"]},
                          "sequence": ["A", "B"]},
                         sort_keys=True, default=str)
    assert sig == hashlib.sha256(payload.encode()).hexdigest()[:12]
    repo = PatternRepository()
    pid = repo.add(dict(p))
    pid2 = repo.add({"type": "sequential", "features": {"ngram": ["A", "B"]},
                     "sequence": ["A", "B"], "frequency": 2,
                     "occurrences": [5], "confidence": 0.9})
    assert pid2 == pid and repo.get(pid)["frequency"] == 5


def test_repository_update_blocklist():
    repo = PatternRepository()
    pid = repo.add({"type": "t", "features": {"a": 1}, "sequence": ["x"],
                    "frequency": 4, "occurrences": [1], "confidence": 0.5})
    assert repo.update(pid, {"state": "CONFIRMED", "type": "HACK",
                             "frequency": 99, "occurrences": [9],
                             "sequence": ["z"], "features": {"b": 2},
                             "id": "P-999"}) is True
    cur = repo.get(pid)
    assert cur["state"] == "CONFIRMED"
    assert (cur["type"], cur["frequency"], cur["occurrences"],
            cur["sequence"], cur["features"], cur["id"]) == \
        ("t", 4, [1], ["x"], {"a": 1}, pid)
    assert repo.update("P-999", {"state": "X"}) is False


def test_clustering_guards_and_stability():
    vecs = [[float(i)] for i in range(30)] + [[1000.0]]
    regs = find_regimes([v[0] for v in vecs], size=2, k=2, method="dbscan")
    assert all(r["features"]["cluster"] != -1 for r in regs)
    assert all(r["features"]["method"] == "dbscan" for r in regs)
    with pytest.raises(ValueError):
        find_regimes(list(range(1100)), size=2, method="dbscan")
    with pytest.raises(ValueError):
        find_regimes(list(range(1100)), size=2, method="agglomerative")
    r = kmeans(_blobs(), 2)
    assert r["labels"][0] == r["labels"][1] and r["labels"][0] != r["labels"][3]
    a = agglomerative(_blobs(), 2)
    assert a["labels"][0] == a["labels"][1] and len(a["merges"]) == 4


# ---------- report limit ----------

def test_report_limit_param():
    res = {"patterns": [{"id": f"P-{i:03d}", "type": "t", "frequency": 1,
                         "confidence": 0.5, "state": "NEW", "features": {},
                         "occurrences": []} for i in range(5)],
           "matches": [{"pattern_id": f"P-{i:03d}", "similarity": 0.9}
                       for i in range(5)],
           "anomalies": [{"index": i, "value": i, "kind": "k",
                          "score": 0.9, "explanation": "e"} for i in range(5)],
           "predictions": [{"next": "A", "probability": 0.5,
                            "evidence": "e"} for _ in range(5)]}
    md1 = render_markdown(res, limit=2)
    assert md1.count("## P-") == 2
    assert "+3 more pattern(s)" in md1
    assert "+3 more patterns" in render_text(res, limit=2)
    assert "P-004" in render_markdown(res)
    assert "P-004" in render_text(res)


# ---------- arithmetic reorder ----------

def test_arithmetic_poly_first():
    r = analyze_numeric_sequence([5, 5, 5, 5])
    assert r["kind"] == "arithmetic" and r["next"] == [5]
    r2 = analyze_numeric_sequence([2, 2, 4])
    assert r2["kind"] == "fibonacci_like" and r2["next"][0] == 6


# ---------- engine wiring ----------

def test_engine_score_detail_on_patterns():
    disc = Nexora(config={"min_support": 2}).discover(list("ABCABC"))
    assert disc["patterns"]
    assert all("score_detail" in p for p in disc["patterns"])
    assert all("confidence" in p["score_detail"] for p in disc["patterns"])


def test_engine_new_patterns_subset():
    nx = Nexora(config={"min_support": 2})
    d1 = nx.discover(list("ABCABC"))
    assert set(p["id"] for p in d1["new_patterns"]) <= {p["id"] for p in d1["patterns"]}
    assert d1["new_patterns"]  # first run creates everything
    d2 = nx.discover(list("ABCABC"))
    assert d2["new_patterns"] == []  # re-run merges, nothing new


def test_engine_skipped_evidence():
    disc = Nexora().discover([1.0, 2.0])
    sk = disc["evidence"]["skipped"]
    assert "regimes" in sk and "seasonality" in sk and "correlation" in sk
    assert "16" in sk["regimes"] and "multivariate" in sk


def test_engine_structural_evidence():
    disc = Nexora(config={"min_support": 2}).discover(list("ABCABCABC"))
    assert disc["evidence"]["structural"]["nodes"] == 3
    # NOTE (v2/WS2): on pure ABC repeats every association rule is
    # subsumed by the closed ABC pattern with equal support, so none is
    # stored (D5). A rule that adds information beyond the sequences —
    # A,B co-occurring across chunks without a frequent AB bigram —
    # is still stored:
    disc2 = Nexora(config={"min_support": 2}).discover(list("AXBAYB"))
    assoc = [p for p in disc2["patterns"] if p["type"] == "association"]
    assert assoc and all(0.0 <= p["confidence"] <= 1.0 for p in assoc)


def test_predict_next_sources():
    nx = Nexora()
    a = nx.predict_next([2, 4, 6, 8])
    assert a["source"] == "arithmetic" and a["next"] == 10 and a["probability"] == 1.0
    c = nx.predict_next(list("ABCABCABC"))
    assert c["source"] in ("context", "markov") and c["next"] == "A" and c["probability"] == 1.0
    n = nx.predict_next([])
    assert n == {"next": None, "probability": 0.0, "source": "none",
                 "evidence": "no recorded transitions"}
    m = build_transition_matrix(["A", "B", "A"])
    assert predict_next("A", m)[0]["next"] == "B"
    cm = build_context_model(list("ABCABC"), max_order=2)
    assert predict_with_context(cm, ["A", "B"])[0]["next"] == "C"


def test_trend_forecast_present_on_rising_data():
    assert detect_trend([1, 2, 3, 4, 5])["direction"] == "rising"
    pr = Nexora().predict([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pr["extrapolation"]["trend"]["direction"] == "rising"
    assert pr["extrapolation"]["trend"]["next"][0] == pytest.approx(6.0)


def test_max_period_respected():
    sig = [math.sin(2 * math.pi * i / 4) for i in range(40)]
    est = estimate_period(sig, max_period=6)
    assert est["period"] is None or est["period"] <= 6
    cfg = Nexora(config={"max_period": 6}).config
    assert cfg["max_period"] == 6


def test_multivariate_end_index_flagged():
    # NOTE (v2/D3): this test previously asserted the buggy behavior —
    # index 30 reported twice (multivariate z=0.0 + statistical) for a
    # univariate series. Per D3 a univariate series must not go through
    # the multivariate detector and one event yields one record.
    out = Nexora().find_anomalies([10.0] * 30 + [25.0])
    idxs = [a["index"] for a in out["anomalies"]]
    assert max(idxs) == 30
    assert idxs == [30]
    # NOTE (v2/WS4): robust joins statistical in the single merged record.
    assert "statistical" in out["anomalies"][0]["kind"]


def test_quality_still_fine():
    assert quality_report([1, 2, 3, 4, 5])["quality"] == pytest.approx(1.0)
    assert Nexora().quality([1, 2, 3, 4, 5])["quality"] == pytest.approx(1.0)
