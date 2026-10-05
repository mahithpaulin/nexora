"""Seeded synthetic datasets for the Nexora evaluation harness.

Stdlib only. Every generator takes a ``seed`` and uses ``random.Random(seed)``
as its sole randomness source, so the same seed always yields identical
``(data, truth)`` pairs. ``truth`` is always plain JSON-able dicts/lists.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple


def periodic(
    n: int = 300,
    period: Tuple[str, ...] = ("A", "B", "C"),
    seed: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    """Exact repetition of ``period`` (``seed`` kept for interface uniformity)."""
    plist = list(period)
    data = [plist[i % len(plist)] for i in range(n)]
    truth = {"type": "periodic", "period": plist, "anomalies": []}
    return data, truth


def drifting(n: int = 300, seed: int = 1) -> Tuple[List[float], Dict[str, Any]]:
    """Numeric random walk with a small constant linear drift."""
    rng = random.Random(seed)
    slope = 0.05
    data: List[float] = []
    x = 0.0
    for _ in range(n):
        x += slope + (rng.random() - 0.5)
        data.append(x)
    truth: Dict[str, Any] = {"type": "drift", "drift": slope, "anomalies": []}
    return data, truth


def regime_change(
    n: int = 300, change_at: int = 150, seed: int = 2
) -> Tuple[List[float], Dict[str, Any]]:
    """Numeric series with mean 0 before ``change_at`` and mean 5 after."""
    rng = random.Random(seed)
    data = [
        (0.0 if i < change_at else 5.0) + (rng.random() - 0.5) * 2.0
        for i in range(n)
    ]
    truth: Dict[str, Any] = {"type": "regime_change", "change_at": change_at}
    return data, truth


def noisy_cycle(
    n: int = 300, corrupt_frac: float = 0.2, seed: int = 3
) -> Tuple[List[str], Dict[str, Any]]:
    """ABC cycle with a seeded fraction of positions replaced by junk tokens."""
    rng = random.Random(seed)
    period = ["A", "B", "C"]
    data: List[str] = []
    for i in range(n):
        if rng.random() < corrupt_frac:
            data.append(rng.choice(["X", "Y", "Z"]))
        else:
            data.append(period[i % len(period)])
    truth: Dict[str, Any] = {
        "type": "periodic",
        "period": ["A", "B", "C"],
        "anomalies": [],
    }
    return data, truth


def planted_anomalies(
    n: int = 300,
    spikes: Dict[int, float] = {150: 25.0, 220: -20.0},  # noqa: B006
    seed: int = 4,
) -> Tuple[List[float], Dict[str, Any]]:
    """~N(0, 1) baseline via Box-Muller on ``random.Random`` with spikes planted."""
    rng = random.Random(seed)
    spikes = dict(spikes)
    data: List[float] = []
    for _ in range(n):
        u1 = rng.random()
        while u1 == 0.0:
            u1 = rng.random()
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data.append(z)
    for idx, val in spikes.items():
        data[idx] = float(val)
    truth: Dict[str, Any] = {
        "type": "spikes",
        "anomaly_indices": sorted(spikes),
    }
    return data, truth


def noise(
    n: int = 300, alphabet: int = 20, seed: int = 5
) -> Tuple[List[str], Dict[str, Any]]:
    """Uniform random tokens from a fixed alphabet (no planted structure)."""
    rng = random.Random(seed)
    data = ["T%d" % rng.randrange(alphabet) for _ in range(n)]
    truth: Dict[str, Any] = {"type": "noise", "anomalies": []}
    return data, truth
