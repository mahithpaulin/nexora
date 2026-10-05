"""Smoke tests for the WORKSTREAM 9 evaluation harness (bench/ only)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.datasets import (  # noqa: E402
    drifting,
    noisy_cycle,
    noise,
    periodic,
    planted_anomalies,
    regime_change,
)
from bench.floors import FLOORS  # noqa: E402
from bench.metrics import f1, precision_recall_f1  # noqa: E402


def test_datasets_deterministic_same_seed():
    assert periodic(seed=0) == periodic(seed=0)
    assert drifting(seed=1) == drifting(seed=1)
    assert regime_change(seed=2) == regime_change(seed=2)
    assert noisy_cycle(seed=3) == noisy_cycle(seed=3)
    assert planted_anomalies(seed=4) == planted_anomalies(seed=4)
    assert noise(seed=5) == noise(seed=5)
    # different seeds diverge somewhere
    assert noisy_cycle(seed=3)[0] != noisy_cycle(seed=99)[0]
    assert noise(seed=5)[0] != noise(seed=99)[0]


def test_datasets_truth_shapes():
    _, t = periodic()
    assert t == {"type": "periodic", "period": ["A", "B", "C"], "anomalies": []}
    _, t = drifting()
    assert t["type"] == "drift" and isinstance(t["drift"], float)
    assert t["anomalies"] == []
    _, t = regime_change()
    assert t == {"type": "regime_change", "change_at": 150}
    _, t = noisy_cycle()
    assert t == {"type": "periodic", "period": ["A", "B", "C"], "anomalies": []}
    data, t = planted_anomalies()
    assert t == {"type": "spikes", "anomaly_indices": [150, 220]}
    assert data[150] == 25.0 and data[220] == -20.0
    _, t = noise()
    assert t == {"type": "noise", "anomalies": []}
    for fn in (periodic, drifting, regime_change, noisy_cycle,
               planted_anomalies, noise):
        data, truth = fn()
        assert len(data) == 300
        json.dumps(truth)  # truth must be plain JSON-able


def test_metrics_exact_on_toy():
    # Genuine 0.5 case: tp=1, fp=1, fn=1.
    r = precision_recall_f1({1, 2}, {2, 3})
    assert r == {
        "tp": 1.0,
        "fp": 1.0,
        "fn": 1.0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    # {1,2,3} vs {2,3,4} has tp=2, so the exact scores are 2/3, not 0.5.
    r2 = precision_recall_f1({1, 2, 3}, {2, 3, 4})
    assert r2["tp"] == 2.0 and r2["fp"] == 1.0 and r2["fn"] == 1.0
    assert r2["precision"] == 2.0 / 3.0
    assert r2["recall"] == 2.0 / 3.0
    assert r2["f1"] == 2.0 / 3.0
    assert f1({1, 2}, {2, 3}) == 0.5
    assert f1({1, 2, 3}, {2, 3, 4}) == 2.0 / 3.0
    # edge cases: both empty is perfect; one-sided empties score 0 rates
    assert precision_recall_f1([], [])["f1"] == 1.0
    assert precision_recall_f1([], [1])["f1"] == 0.0
    assert precision_recall_f1([1], [])["f1"] == 0.0


def test_run_bench_imports_and_uses_public_api_only():
    import bench.run_bench as rb

    assert hasattr(rb, "main") and hasattr(rb, "run_all")
    with open(os.path.join(os.path.dirname(__file__), "..", "bench",
                           "run_bench.py")) as fh:
        src = fh.read()
    assert "from nexora import Nexora" in src
    assert "from nexora." not in src
    assert "import nexora." not in src


def test_floors_keys_nonempty():
    assert isinstance(FLOORS, dict) and len(FLOORS) > 0
    assert all(isinstance(v, (int, float)) and v >= 0.0
               for v in FLOORS.values())
