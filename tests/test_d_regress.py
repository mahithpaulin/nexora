"""Regression tests for known defects D1-D4 + D6 (v2 bug-fix pass).

Each test reproduces the defect report before asserting the fixed
behavior. D5 (redundant patterns) is covered by the pruning tests in
tests/test_prune.py (workstream 2).
"""
import math

from nexora import Nexora


# ---------- D1: no meaningless mean for categorical data ----------

def test_d1_no_mean_for_categorical():
    out = Nexora().discover(list("ABCABCABC"))
    expl = out["explanation"]
    assert "mean" not in expl
    assert "mode 'A'" in expl  # 3 of 9, first most-common
    assert "3 distinct" in expl


def test_d1_mean_present_for_numeric():
    out = Nexora().discover([1.0, 2.0, 3.0] * 4)
    expl = out["explanation"]
    assert "mean 2.000" in expl
    assert "12 numeric value(s)" in expl


def test_d1_categorical_entropy_cited():
    out = Nexora().discover(list("AAB"))
    expl = out["explanation"]
    # p(A)=2/3, p(B)=1/3 -> entropy ~0.918 bits
    assert "entropy 0.918 bits" in expl


# ---------- D2: no novelty flags on flat series ----------

def test_d2_no_novelty_after_unrelated_discover():
    # Exact D2 repro: README flow shares one engine across calls.
    nx = Nexora()
    nx.discover(list("ABCABCABC"))
    out = nx.find_anomalies([10.0] * 30 + [25.0])
    seq = [a for a in out["anomalies"]
           if a["kind"] in ("novel_sequence", "missing_transition")]
    assert seq == []
    assert all(a["index"] != 1 for a in out["anomalies"])


def test_d2_flat_series_never_novel_even_with_relevant_patterns():
    nx = Nexora(config={"min_support": 2})
    nx.discover([10.0] * 10)  # patterns share the numeric vocabulary
    out = nx.find_anomalies([10.0] * 40)
    seq = [a for a in out["anomalies"]
           if a["kind"] in ("novel_sequence", "missing_transition")]
    assert seq == []


def test_d2_sequence_z_is_none_not_zero():
    # A z of 0.0 implies a measured z-score; sequence hits measure none.
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABCABCABC"))
    out = nx.find_anomalies(list("ABZ"))
    mt = [a for a in out["anomalies"] if a["kind"] == "missing_transition"]
    assert mt and all(a["z"] is None for a in mt)


# ---------- D3: one univariate event = one record, no multivariate ----------

def test_d3_univariate_single_record():
    out = Nexora().find_anomalies([10.0] * 30 + [25.0])
    anoms = out["anomalies"]
    assert [a["index"] for a in anoms] == [30]
    assert len(anoms) == 1
    assert anoms[0]["kind"] == "statistical"
    assert "multivariate" not in anoms[0]["kind"]
    assert math.isclose(anoms[0]["z"], 5.477, rel_tol=1e-3)


def test_d3_multivariate_runs_on_multivariate_input():
    rows = [{"value": 10.0, "aux": 1.0}] * 30 + [{"value": 25.0, "aux": 50.0}]
    out = Nexora().find_anomalies(rows)
    at30 = [a for a in out["anomalies"] if a["index"] == 30]
    assert len(at30) == 1  # merged, not reported twice
    assert "multivariate" in at30[0]["kind"]
    assert "statistical" in at30[0]["kind"]
    assert at30[0]["z"] is not None  # statistical z survives the merge


# ---------- D4: explanation generated from the actual decision ----------

def test_d4_why_matches_verdict():
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABCABC"))
    for m in nx.match("Z"):
        if not m["matched"]:
            assert "did not match" in m["explanation"]
            assert "matched features" not in m["explanation"]
        else:  # pragma: no cover - Z matches nothing here
            assert "matched pattern" in m["explanation"]
    good = [m for m in nx.match("A") if m["matched"]]
    assert good
    assert all("matched pattern" in m["explanation"] for m in good)
    assert all("did not match" not in m["explanation"] for m in good)


def test_d4_only_real_overlapping_features_cited():
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABCABC"))
    misses = [m for m in nx.match("Z") if not m["matched"]]
    assert misses
    assert all("none" in m["explanation"] for m in misses)


# ---------- D6: single accurate README, no stale repo details ----------

def test_d6_readme_single_version_section():
    with open("README.md") as fh:
        text = fh.read()
    assert "v0.1" not in text
    assert "v0.2-dev" not in text
    assert "Opencode bot" not in text
    assert text.count("## ") <= 3


def test_d6_pyproject_description_current():
    with open("pyproject.toml") as fh:
        text = fh.read()
    assert "v0.1" not in text
    assert "Nexora — non-neural" in text
