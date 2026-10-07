"""Workstream 4: robust anomaly detection.

Median/MAD rolling scores, level shifts, seasonality-aware residuals,
one merged record per event, severity levels. No false positives on
constant data.
"""
import math

from nexora import Nexora
from nexora.anomaly.robust import (level_shift_records, mad_of, modified_z,
                                   robust_detect, severity_of)


def test_severity_mapping():
    assert severity_of(0.0) == "low"
    assert severity_of(0.49) == "low"
    assert severity_of(0.5) == "medium"
    assert severity_of(0.69) == "medium"
    assert severity_of(0.7) == "high"
    assert severity_of(0.89) == "high"
    assert severity_of(0.9) == "critical"
    assert severity_of(1.0) == "critical"
    assert severity_of("junk") == "low"


def test_modified_z_basics():
    assert modified_z(10.0, 10.0, 0.0) == 0.0
    assert modified_z(11.0, 10.0, 0.0) == math.inf
    assert abs(modified_z(11.0, 10.0, 1.0) - 0.6745) < 1e-9
    assert mad_of([1, 1, 1, 1]) == 0.0
    assert mad_of([1, 2, 3]) == 1.0


def test_spike_merged_single_record_with_severity():
    out = Nexora().find_anomalies([10.0] * 30 + [25.0])
    assert len(out["anomalies"]) == 1
    a = out["anomalies"][0]
    assert a["index"] == 30
    assert "statistical" in a["kind"] and "robust" in a["kind"]
    assert a["severity"] in ("high", "critical")
    assert a["z"] is not None  # statistical z survives the merge


def test_level_shift_detected():
    out = Nexora().find_anomalies([1.0] * 20 + [10.0] * 20)
    ls = [a for a in out["anomalies"] if "level_shift" in a["kind"]]
    assert ls
    assert any(15 <= a["index"] <= 25 for a in ls)
    assert all(a["z"] is None for a in ls)  # windowed shift, not a point z
    assert all(a["severity"] in ("medium", "high", "critical") for a in ls)


def test_level_shift_direct_module():
    recs = level_shift_records([1.0] * 20 + [10.0] * 20, window=10, threshold_z=3.0)
    assert recs and all(r["kind"] == "level_shift" for r in recs)
    assert level_shift_records([5.0] * 40, window=10) == []
    assert level_shift_records([1.0, 2.0], window=10) == []  # too short


def test_flat_series_no_false_positives():
    out = Nexora().find_anomalies([7.0] * 40)
    assert out["anomalies"] == []


def test_short_jump_flagged_by_robust_not_forced():
    # A 99x jump from a perfectly flat past is genuinely anomalous;
    # the rolling MAD detector flags it (score high), the global z
    # cannot (scales with sqrt(n)). One record, honestly sourced.
    out = Nexora().find_anomalies([1.0, 1.0, 1.0, 99.0])
    assert len(out["anomalies"]) == 1
    assert out["anomalies"][0]["kind"] == "robust"
    assert out["anomalies"][0]["z"] == math.inf


def test_seasonal_residual_unmasks_spike():
    base = [0.0, 10.0] * 20
    data = base + [25.0]
    plain = robust_detect(data, window=20, threshold=3.5)
    # Honest limitation documented: against a 3-point trailing history a
    # period-2 oscillation looks like constant jumping (MAD=0), so the
    # plain detector fires repeatedly on the cycle itself.
    assert len(plain) > 5
    adj = robust_detect(data, window=20, threshold=3.5, period=2)
    assert [r["index"] for r in adj] == [40]
    assert "deseasonalized(period=2)" in adj[0]["causes"][0]


def test_robust_is_causal_and_deterministic():
    data = [float(i % 7) for i in range(60)] + [100.0]
    r1 = robust_detect(data, window=10, threshold=3.0)
    r2 = robust_detect(data, window=10, threshold=3.0)
    assert r1 == r2
    assert r1 and r1[-1]["index"] == 60
    # The spike at 60 was scored without seeing itself: same history,
    # same verdict even if later points change.
    r3 = robust_detect(data + [200.0, 300.0], window=10, threshold=3.0)
    assert [r["index"] for r in r3 if r["index"] == 60] == [60]


def test_robust_skips_junk():
    rows = [{"index": i, "value": v} for i, v in
            enumerate([1.0, None, float("nan"), True, 1.0, 1.0, 50.0])]
    recs = robust_detect(rows, window=3, threshold=2.0)
    assert all(isinstance(r["index"], int) for r in recs)


def test_robust_bad_args_raise():
    for bad in (lambda: robust_detect([1.0], window=0),
                lambda: robust_detect([1.0], threshold=0),
                lambda: robust_detect([1.0], threshold=-1)):
        try:
            bad()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_config_flags_disable_paths():
    nx = Nexora(config={"robust": False, "level_shifts": False})
    out = nx.find_anomalies([10.0] * 30 + [25.0])
    assert [a["kind"] for a in out["anomalies"]] == ["statistical"]
    nx2 = Nexora(config={"level_shifts": False})
    out2 = nx2.find_anomalies([1.0] * 20 + [10.0] * 20)
    assert all("level_shift" not in a["kind"] for a in out2["anomalies"])
