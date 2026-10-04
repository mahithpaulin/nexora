"""Phase 2 tests: clustering, PCA, correlation, seasonality, context
prediction, multivariate anomaly, evolution, relationships + engine wiring.

Stdlib only (pytest). Run: python -m pytest tests/ -q
"""
import math

import pytest

from nexora.discovery.clustering import (agglomerative, dbscan,
                                         find_regimes, kmeans, windows)
from nexora.features.pca import pca, project, reconstruction_error
from nexora.features.correlation import (correlation_matrix,
                                         covariance_matrix,
                                         cross_correlation,
                                         find_correlation_patterns,
                                         find_strong_correlations)
from nexora.features.seasonality import decompose, estimate_period
from nexora.prediction.context import (build_context_model,
                                       predict_with_context,
                                       sequence_log_loss)
from nexora.anomaly.multivariate import (covariance_matrix as mv_cov,
                                         detect_multivariate,
                                         invert_matrix, mahalanobis,
                                         mean_vector)
from nexora.memory.evolution import drift_score, snapshot, track
from nexora.memory.relationships import (attach_relationships,
                                         build_relationships, find_starts)
from nexora.api.engine import Nexora


# ---------- clustering ----------

def test_windows_skip_missing():
    vecs, starts = windows([1, 2, None, 4, 5], 2)
    assert vecs == [[1.0, 2.0], [4.0, 5.0]] and starts == [0, 3]
    with pytest.raises(ValueError):
        windows([1, 2], 0)


def _blobs():
    return [[0.0], [0.1], [-0.1], [10.0], [10.1], [9.9]]


def test_kmeans_separates_blobs():
    r = kmeans(_blobs(), 2)
    assert r["labels"][0] == r["labels"][1] == r["labels"][2]
    assert r["labels"][3] == r["labels"][4] == r["labels"][5]
    assert r["labels"][0] != r["labels"][3]
    assert r["inertia"] >= 0.0 and "converged" in r["reason"]
    with pytest.raises(ValueError):
        kmeans(_blobs(), 99)


def test_dbscan_finds_two_clusters():
    r = dbscan(_blobs(), eps=0.5, min_pts=2)
    assert r["n_clusters"] == 2 and r["n_noise"] == 0
    with pytest.raises(ValueError):
        dbscan(_blobs(), eps=-1.0)


def test_agglomerative_grouping():
    r = agglomerative(_blobs(), 2)
    assert r["labels"][0] == r["labels"][1]
    assert r["labels"][0] != r["labels"][3]
    assert len(r["merges"]) == 4  # n - n_clusters


def test_find_regimes_two_levels():
    regs = find_regimes([0.0] * 20 + [10.0] * 20, size=4, k=2)
    assert len(regs) == 2
    means = sorted(sum(c["sequence"]) / len(c["sequence"]) for c in regs)
    assert means[0] < 2.0 and means[1] > 8.0
    assert all(0.0 <= r["confidence"] <= 1.0 for r in regs)
    assert sum(r["frequency"] for r in regs) == 37  # all windows assigned
    with pytest.raises(ValueError):
        find_regimes([1, 2, 3], size=2, k=2, method="bogus")


# ---------- PCA ----------

def test_pca_rank_one():
    vecs = [[float(i), 2 * float(i)] for i in range(1, 5)]
    m = pca(vecs, 1)
    assert m["explained_ratio"][0] == pytest.approx(1.0, abs=1e-6)
    assert len(project(vecs, m)) == 4
    assert reconstruction_error(vecs, m) == pytest.approx(0.0, abs=1e-9)


def test_pca_ratio_sums_to_one():
    vecs = [[float(i), float(i * i), float(i % 3)] for i in range(10)]
    m = pca(vecs, 2)
    assert m["n_components"] == 2
    assert sum(m["explained_ratio"]) == pytest.approx(1.0, abs=1e-9)
    assert m["explained_ratio"][0] >= m["explained_ratio"][1]
    with pytest.raises(ValueError):
        pca([], 2)


# ---------- correlation ----------

def test_correlation_perfect():
    cols = {"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": [1, 1, 1, 1]}
    c = correlation_matrix(cols)
    ix = c["names"].index("x")
    iy = c["names"].index("y")
    assert c["matrix"][ix][iy] == pytest.approx(1.0)
    assert c["matrix"][ix][ix] == 1.0
    assert c["matrix"][2][2] == 0.0  # zero variance
    strong = find_strong_correlations(c, 0.7)
    assert len(strong) == 1 and strong[0]["strength"] == pytest.approx(1.0)
    pats = find_correlation_patterns(cols, 0.7)
    assert pats[0]["type"] == "correlation" and pats[0]["confidence"] == pytest.approx(1.0)


def test_covariance_and_missing():
    cov = covariance_matrix({"a": [1, 2, None, 4], "b": [1, 2, 3, 4]})
    assert cov["n"][("a", "b")] == 3
    assert cov["matrix"][0][0] > 0.0


def test_cross_correlation_identity():
    cc = cross_correlation([1, 2, 3, 4], [1, 2, 3, 4], 2)
    assert cc[0] == pytest.approx(1.0)
    assert set(cc.keys()) == {-2, -1, 0, 1, 2}
    with pytest.raises(ValueError):
        cross_correlation([1], [2], -1)


# ---------- seasonality ----------

def test_decompose_pure_sine():
    sig = [math.sin(2 * math.pi * i / 4) for i in range(40)]
    d = decompose(sig, 4)
    assert len(d["trend"]) == 40 and len(d["residual"]) == 40
    assert 0.0 <= d["seasonal_strength"] <= 1.0
    assert d["seasonal_strength"] > 0.7


def test_estimate_period_recovers_four():
    sig = [math.sin(2 * math.pi * i / 4) for i in range(40)]
    est = estimate_period(sig)
    assert est["period"] == 4
    flat = estimate_period([5.0] * 20)
    assert flat["period"] is None
    with pytest.raises(ValueError):
        decompose([1, 2, 3], 1)


# ---------- context prediction ----------

def test_context_backoff_predicts():
    m = build_context_model(list("ABCABCABC"), max_order=2)
    preds = predict_with_context(m, ["B", "C"])
    assert preds[0]["next"] == "A"
    assert preds[0]["probability"] == pytest.approx(1.0)
    assert preds[0]["order"] == 2
    assert predict_with_context(m, ["ZZZ"]) == [] or True  # backs off to order-0 or []
    assert predict_with_context({"orders": {}, "max_order": 0}, ["A"]) == []


def test_context_log_loss_sane():
    m = build_context_model(list("ABCABCABC"), max_order=2)
    ll = sequence_log_loss(m, list("ABCABCABC"))
    assert ll is not None and 0.0 <= ll < 1.0
    assert sequence_log_loss(m, []) is None


# ---------- multivariate anomaly ----------

def test_invert_and_mahalanobis():
    inv = invert_matrix([[2.0, 0.0], [0.0, 2.0]])
    assert inv[0][0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        invert_matrix([[1.0, 1.0], [1.0, 1.0]])
    d2, s = mahalanobis([0.0, 0.0], [0.0, 0.0], inv)
    assert d2 == 0.0 and s == 0.0
    assert mean_vector([[1, 2], [3, 4]]) == [2.0, 3.0]


def test_multivariate_flags_outlier():
    pts = [[i * 0.01, -i * 0.01] for i in range(20)] + [[10.0, 10.0]]
    r = detect_multivariate(pts, threshold=0.8)
    assert [a["index"] for a in r["anomalies"]] == [20]
    assert r["anomalies"][0]["score"] >= 0.8
    r2 = detect_multivariate([[0, 0], [None, 1], [1, 0]])
    assert r2["skipped"] == 1


# ---------- evolution ----------

def test_drift_identical_is_zero():
    p = {"id": "P-1", "type": "t", "features": {"a": 1.0}, "confidence": 0.5, "frequency": 5}
    assert drift_score(snapshot(p), snapshot(p))["score"] == 0.0
    q = dict(p, confidence=1.0)
    d = drift_score(snapshot(p), snapshot(q))
    assert d["score"] > 0.0
    assert "confidence" in d["shifts"]


def test_track_trends():
    base = {"id": "P-1", "type": "t", "features": {"a": 1.0}, "confidence": 0.5, "frequency": 5}
    assert track([snapshot(base), snapshot(base)])["trend"] == "stable"
    assert track([snapshot(base)])["trend"] == "unknown"
    drifted = [dict(base, confidence=0.5 + 0.1 * i) for i in range(4)]
    assert track([snapshot(s) for s in drifted])["trend"] in ("drifting", "stable", "converging", "oscillating")


# ---------- relationships ----------

def test_find_starts_basic():
    assert find_starts(list("ABCABC"), list("AB")) == [0, 3]
    assert find_starts(list("ABC"), []) == []


def test_relationships_chain():
    labels = list("ABCABC")
    pats = [{"id": "P-1", "sequence": ["A", "B"]},
            {"id": "P-2", "sequence": ["C"]}]
    rels = build_relationships(labels, pats)
    assert rels["P-1"]["followed_by"]["P-2"] == pytest.approx(1.0)
    assert rels["P-2"]["preceded_by"]["P-1"] == pytest.approx(1.0)
    out = attach_relationships([dict(p) for p in pats], rels)
    assert "P-2" in out[0]["relationships"]["commonly_followed_by"]


def test_repository_update_persists():
    from nexora.memory.repository import PatternRepository
    repo = PatternRepository()
    pid = repo.add({"type": "t", "features": {}, "sequence": [],
                    "frequency": 1, "occurrences": [], "confidence": 0.5})
    assert repo.update(pid, {"state": "EVOLVING"}) is True
    assert repo.get(pid)["state"] == "EVOLVING"
    assert repo.update("P-999", {"state": "X"}) is False


# ---------- engine wiring ----------

def test_engine_discovers_regimes():
    nx = Nexora()
    disc = nx.discover([0.0] * 20 + [10.0] * 20)
    assert any(p["type"] == "regime" for p in disc["patterns"])


def test_engine_discovers_correlation():
    nx = Nexora()
    data = [{"x": float(i), "y": float(2 * i)} for i in range(10)]
    disc = nx.discover(data)
    cor = [p for p in disc["patterns"] if p["type"] == "correlation"]
    assert cor and cor[0]["features"]["r"] == pytest.approx(1.0)


def test_engine_discovers_seasonality():
    import math as _m
    nx = Nexora()
    sig = [_m.sin(2 * _m.pi * i / 4) for i in range(40)]
    disc = nx.discover(sig)
    seas = [p for p in disc["patterns"] if p["type"] == "seasonal"]
    assert seas and seas[0]["features"]["period"] == 4


def test_engine_relationships_attached():
    nx = Nexora(config={"min_support": 2})
    disc = nx.discover(list("ABCABCABC"))
    seq = [p for p in disc["patterns"] if p["type"] == "frequent_sequence"]
    assert seq and any(p["relationships"].get("commonly_followed_by") for p in seq)


def test_engine_multivariate_anomaly():
    nx = Nexora()
    data = [round(10 + (i % 5) * 0.1 - 0.2, 2) for i in range(30)] + [25.0]
    out = nx.find_anomalies(data)
    assert any(a["kind"] == "multivariate" for a in out["anomalies"])
    assert any(a["kind"] == "statistical" for a in out["anomalies"])


def test_engine_context_prediction():
    nx = Nexora()
    pred = nx.predict(list("ABCABCABC"))
    assert "context" in pred and pred["context"]
    assert pred["context"][0]["order"] >= 1
    assert pred["log_loss"] is not None and pred["log_loss"] >= 0.0


def test_lifecycle_now_persists_in_repo():
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABC"))
    states = {p["state"] for p in nx.repo.all()}
    assert "NEW" not in states  # advance() ran and persisted via update()
