"""Minimum acceptable bench scores, keyed by ``"<dataset>/<metric>"``.

Calibrated 2026-10-05 on the v2 engine (all datasets seeded, so scores
are deterministic — floors equal measured values). ``bench/run_bench.py``
exits nonzero if any reported metric falls below its floor. Recalibrate
deliberately when behavior changes; never lower a floor to silence a
regression. The baseline floor is a sanity bar (task nontriviality),
not an engine quality bar.
"""

from __future__ import annotations

FLOORS = {
    "periodic/discovery_precision": 1.0,
    "periodic/discovery_recall": 1.0,
    "periodic/discovery_f1": 1.0,
    "noise/discovery_f1": 1.0,
    "planted_anomalies/anomaly_f1": 1.0,
    "periodic/predict_accuracy": 1.0,
    "periodic/baseline_accuracy": 0.2,
}
