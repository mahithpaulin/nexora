"""Nexora Core v0.1 tests — synthetic datasets with known ground truth.

Stdlib only (pytest). Every algorithm has at least one test here.
Run: python -m pytest tests/ -q
"""
import math

import pytest

from nexora.core.observation import normalize_observations, validate_observations
from nexora.core.pattern import Pattern, make_pattern_id
from nexora.core.scoring import score_pattern
from nexora.ingestion.parser import parse
from nexora.ingestion.validator import validate


# ---------- observation ----------

def test_normalize_mixed_types():
    rows = normalize_observations([1.0, "A", None, {"value": 5, "label": "x"}])
    assert len(rows) == 4
    assert rows[0]["value"] == 1.0
    assert rows[1]["label"] == "A"
    assert rows[2]["value"] is None
    assert rows[3]["value"] == 5 and rows[3]["label"] == "x"


def test_normalize_none_input_empty():
    assert normalize_observations(None) == []


def test_validate_empty_fails():
    with pytest.raises(ValueError):
        validate_observations([])


def test_validate_ok():
    rows = normalize_observations([1, 2, 3])
    assert validate_observations(rows) == rows


# ---------- pattern ----------

def test_pattern_ids_sequential():
    assert make_pattern_id(1) == "P-001"
    assert make_pattern_id(17) == "P-017"


def test_pattern_roundtrip():
    p = Pattern(id="P-017", type="sequential", frequency=847, confidence=0.93)
    q = Pattern.from_dict(p.to_dict())
    assert q.id == "P-017" and q.frequency == 847 and q.confidence == 0.93


def test_pattern_scores_clamped():
    p = Pattern(confidence=5.0, novelty=-1.0)
    assert p.confidence == 1.0 and p.novelty == 0.0


# ---------- scoring ----------

def test_scoring_components_inspectable():
    s = score_pattern(8, 10, 0.9, 0.91, 0.7, 0.78, 0.1, 0.05)
    for k in ("confidence", "frequency_score", "similarity_score",
              "stability_score", "novelty_score", "predictive_score"):
        assert 0.0 <= s[k] <= 1.0
    assert abs(s["frequency_score"] - 0.8) < 1e-9
    assert "explanation" in s and isinstance(s["explanation"], str)


def test_scoring_noise_reduces_confidence():
    clean = score_pattern(8, 10, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0)
    noisy = score_pattern(8, 10, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
    assert noisy["confidence"] < clean["confidence"]


def test_scoring_missing_inputs_safe():
    s = score_pattern(None, 0, None, None, None, None, None, None)
    assert 0.0 <= s["confidence"] <= 1.0


# ---------- ingestion ----------

def test_parse_json_and_csv():
    rows = parse('[1, 2, 3]')
    assert [r["value"] for r in rows] == [1, 2, 3]
    rows = parse("value,label\n1,A\n2,B")
    assert rows[0]["value"] == 1 and rows[0]["label"] == "A"


def test_validator_missing_fraction():
    v = validate([1, None, None, None], max_missing_fraction=0.3)
    assert v["ok"] is False and v["missing"] == 3
    v = validate([1, 2, None], max_missing_fraction=0.5)
    assert v["ok"] is True


def test_validator_constant_warns():
    v = validate([7, 7, 7, 7])
    assert v["ok"] is True and any("constant" in w for w in v["warnings"])
