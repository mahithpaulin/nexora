"""Probability calibration helpers (stdlib only, deterministic).

Reported probabilities stay maximum-likelihood estimates (no hidden
smoothing — ``P=1.0`` still means 5/5 observed), but every estimate
now carries its uncertainty honestly:

- ``wilson_interval``: Wilson score 95% interval for a binomial
  proportion — wide on thin evidence, tight on strong evidence.
- ``calibrate``: bundles ``{probability, ci95, count, total}`` so
  callers can abstain when the evidence is too thin to trust.
"""

from __future__ import annotations

import math


def wilson_interval(count, total, z=1.96) -> tuple[float, float]:
    """Wilson score interval for count/total (default 95%, z=1.96).

    Returns ``(lo, hi)`` clamped to [0, 1]. No observations (total
    <= 0) yields ``(0.0, 1.0)`` — total uncertainty, stated as such.
    """
    try:
        k = float(count)
    except (TypeError, ValueError):
        k = 0.0
    try:
        n = float(total)
    except (TypeError, ValueError):
        n = 0.0
    try:
        zf = float(z)
    except (TypeError, ValueError):
        zf = 1.96
    if not n > 0:
        return (0.0, 1.0)
    if k < 0:
        k = 0.0
    if k > n:
        k = n
    p = k / n
    denom = 1.0 + zf * zf / n
    center = (p + zf * zf / (2.0 * n)) / denom
    half = zf * math.sqrt(p * (1.0 - p) / n + zf * zf / (4.0 * n * n)) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    if k <= 0:
        lo = 0.0
    if k >= n:
        hi = 1.0
    return (lo, hi)


def calibrate(count, total, z=1.96) -> dict:
    """Calibrated view of count/total: MLE probability plus uncertainty."""
    try:
        n = float(total)
    except (TypeError, ValueError):
        n = 0.0
    try:
        k = float(count)
    except (TypeError, ValueError):
        k = 0.0
    p = (k / n) if n > 0 else 0.0
    lo, hi = wilson_interval(k, n, z=z)
    return {"probability": max(0.0, min(1.0, p)), "ci95": [lo, hi],
            "count": int(k) if float(int(k)) == k else k,
            "total": int(n) if float(int(n)) == n else n}
