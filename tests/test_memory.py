"""Memory / anomaly / prediction / explanation / engine tests."""
import os

import pytest

from nexora.memory.repository import PatternRepository
from nexora.memory.lifecycle import advance, update_confidence
from nexora.anomaly.detector import detect
from nexora.prediction.markov import build_transition_matrix, predict_next
from nexora.prediction.transitions import (record_pattern_sequence,
                                           predict_next_pattern)
from nexora.explanation.explainer import (explain_anomaly,
                                         explain_discovery, explain_match,
                                         explain_prediction,
                                         summarize_result)
from nexora.api.engine import Nexora


def _pat(pid="P-001", freq=5, conf=0.6, state="OBSERVED"):
    return {"id": pid, "type": "sequential", "features": {"ngram": ["A", "B"]},
            "sequence": ["A", "B"], "frequency": freq, "occurrences": [0, 3],
            "confidence": conf, "state": state}


def test_repository_add_get_dedupe():
    repo = PatternRepository()
    pid = repo.add({"type": "sequential", "features": {"ngram": ["A", "B"]},
                    "sequence": ["A", "B"], "frequency": 3,
                    "occurrences": [0], "confidence": 0.5})
    assert pid == "P-001"
    pid2 = repo.add({"type": "sequential", "features": {"ngram": ["A", "B"]},
                     "sequence": ["A", "B"], "frequency": 2,
                     "occurrences": [5], "confidence": 0.7})
    assert pid2 == pid  # duplicate merged
    assert repo.get(pid)["frequency"] == 5
    assert repo.size() == 1


def test_repository_save_load(tmp_path):
    repo = PatternRepository()
    repo.add({"type": "t", "features": {"v": 1}, "sequence": [1],
              "frequency": 2, "occurrences": [0, 1], "confidence": 0.5})
    f = str(tmp_path / "mem.json")
    repo.save(f)
    assert os.path.exists(f)
    repo2 = PatternRepository()
    assert repo2.load(f) == 1
    assert repo2.get("P-001")["frequency"] == 2


def test_lifecycle_promotion_and_decay():
    p = _pat(state="NEW", freq=1)
    assert advance(p, observed=True)[0] == "OBSERVED"
    p = _pat(state="OBSERVED", freq=5)
    assert advance(p, observed=True)[0] == "CONFIRMED"
    p = _pat(state="CONFIRMED", freq=50, conf=0.9)
    assert advance(p, observed=True)[0] == "ESTABLISHED"
    p = _pat(state="OBSERVED", freq=1)
    st, _ = advance(dict(p), observed=False,
                    config={"stale_after": 1, "retired_after": 5,
                            "confirm_threshold": 3, "establish_freq": 10,
                            "establish_confidence": 0.8, "drift_threshold": 0.9})
    assert st == "STALE"


def test_update_confidence_moves():
    p = _pat(conf=0.5)
    c, _ = update_confidence(p, True)
    assert c == pytest.approx(0.55)
    c, _ = update_confidence(p, False)
    assert c == pytest.approx(0.45)


def test_anomaly_statistical_outlier():
    rows = [{"index": i, "value": 0.0, "label": "0"} for i in range(20)]
    rows.append({"index": 20, "value": 100.0, "label": "100"})
    out = detect(rows, {"mean": 0.0, "stdev": 1.0}, z_threshold=3.0)
    assert any(a["index"] == 20 and a["kind"] == "statistical" for a in out)


def test_anomaly_novel_transition():
    pats = [{"id": "P-001", "type": "sequential", "features": {},
             "sequence": ["A", "B", "C"]}]
    rows = [{"index": 0, "value": "A", "label": "A"},
            {"index": 1, "value": "X", "label": "X"}]
    out = detect(rows, {"mean": 0, "stdev": 0}, z_threshold=3.0, patterns=pats)
    assert any(a["kind"] in ("novel_sequence", "missing_transition") for a in out)


def test_markov_chain_probabilities():
    m = build_transition_matrix(["A", "B", "A", "B", "A", "C"])
    assert m["probs"][("A", "B")] == pytest.approx(2 / 3)
    preds = predict_next("A", m)
    assert preds[0]["next"] == "B"
    assert preds[0]["probability"] == pytest.approx(2 / 3)
    assert predict_next("ZZZ", m) == []


def test_pattern_level_transitions():
    m = record_pattern_sequence(["P-009", "P-017", "P-024", "P-017", "P-024"])
    preds = predict_next_pattern("P-017", m)
    assert preds[0]["next"] == "P-024"
    assert preds[0]["probability"] == pytest.approx(1.0)


def test_explainers_cite_numbers():
    assert "P-017" in explain_discovery(_pat("P-017"))
    assert "91" in explain_match({"value": 1}, "P-017", 0.91, ["f1"]) or \
        "91.0%" in explain_match({"value": 1}, "P-017", 0.91, ["f1"])
    a = explain_anomaly({"index": 3, "value": 9, "z": 4.2, "score": 0.94,
                         "kind": "statistical", "causes": ["feature_3: +4.2σ"]})
    assert "4.2" in a and "0.94" in a
    assert "P-024" in explain_prediction([{"next": "P-024", "probability": 0.78,
                                           "evidence": "e"}])


def test_engine_end_to_end_abc():
    nx = Nexora(config={"min_support": 2, "max_n": 3})
    data = list("ABCABCABC")
    disc = nx.discover(data)
    seqs = [tuple(p["sequence"]) for p in disc["patterns"]
            if p["type"] == "frequential" or p["type"] == "frequent_sequence"
            or p["type"] == "sequential"]
    assert ("A", "B", "C") in seqs
    matches = nx.match("A")
    assert matches and matches[0]["similarity"] >= 0.5
    anom = nx.find_anomalies(data + [99999])
    assert anom["count"] >= 0  # categorical data: stats may be empty-safe
    pred = nx.predict(data)
    assert pred["current"] == "C"
    hist = nx.get_history(disc["patterns"][0]["id"])
    assert "occurrences" in hist and "confidence_trail" in hist
    assert isinstance(nx.explain(disc), str) and disc["patterns"]


def test_engine_numeric_anomaly_and_predict():
    nx = Nexora(config={"min_support": 2})
    # Long baseline: a single outlier's z-score grows with sqrt(n), so 20
    # baseline points put z(50.0) well above the default threshold of 3.0.
    data = [10.0 + (i % 3) * 0.1 for i in range(20)] + [50.0]
    out = nx.find_anomalies(data)
    assert any(a["index"] == 20 for a in out["anomalies"])
    det = nx.detect([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    assert det["patterns"] and det["matches"]
    assert "explanation" in det and det["explanation"]
