"""Change-point detection for numeric and categorical series (stdlib only).

- ``change_points``: sliding-window mean shift on numeric data.
  At each split, ``z = (after_mean - before_mean) / SE`` with pooled
  ``SE = pooled_stdev * sqrt(2/window)``; ``|z| >= threshold_z`` flags a
  candidate, then non-maximum suppression keeps the strongest per window.
  Returns ``[{"index", "before_mean", "after_mean", "z", "reason"}]``
  (``index`` = input position where the "after" window starts;
  ``z = +/-inf`` when the pooled stdev is 0 but means differ).
- ``label_change_points``: positions where a categorical label changes.
- ``detect_runs``: run-length summary ``[{value, start, end, length}]``.

``None``/``NaN`` numerics are skipped (index-mapped back); ``None``
categoricals form their own token (reported explicitly, never silently
dropped). Returns ``[]`` when data are insufficient.
"""

from __future__ import annotations

import math
import statistics


def _clean_indexed(values) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for i, v in enumerate(list(values)):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if not isinstance(v, (int, float)):
            raise ValueError(f"non-numeric value {v!r}")
        out.append((i, float(v)))
    return out


def change_points(values, window=10, threshold_z=3.0) -> list[dict]:
    """Sliding-window mean-shift change points (see module docstring)."""
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive int")
    if threshold_z < 0:
        raise ValueError("threshold_z must be >= 0")
    indexed = _clean_indexed(values)
    m = len(indexed)
    if m < 2 * window:
        return []
    xs = [v for _, v in indexed]
    cands: list[tuple[float, int, float, float]] = []  # (z, split, bmean, amean)
    norm = math.sqrt(2.0 / window)
    for s in range(window, m - window + 1):
        before = xs[s - window: s]
        after = xs[s: s + window]
        bm = sum(before) / window
        am = sum(after) / window
        pooled = statistics.pstdev(before + after)
        se = pooled * norm
        if se == 0.0:
            z = 0.0 if am == bm else (math.inf if am > bm else -math.inf)
        else:
            z = (am - bm) / se
        if abs(z) >= threshold_z:
            cands.append((z, s, bm, am))
    cands.sort(key=lambda t: (-abs(t[0]), t[1]))
    accepted: list[tuple[float, int, float, float]] = []
    for z, s, bm, am in cands:
        if all(abs(s - a[1]) >= window for a in accepted):
            accepted.append((z, s, bm, am))
    accepted.sort(key=lambda t: t[1])
    out = []
    for z, s, bm, am in accepted:
        out.append({"index": indexed[s][0], "before_mean": bm,
                    "after_mean": am, "z": z,
                    "reason": f"mean {bm:.4g} -> {am:.4g} (z={z:.2f})"})
    return out


def label_change_points(labels) -> list[dict]:
    """Indices where the categorical label differs from its predecessor."""
    items = list(labels)
    out: list[dict] = []
    for i in range(1, len(items)):
        if items[i] != items[i - 1]:
            out.append({"index": i, "before": items[i - 1], "after": items[i],
                        "reason": f"label {items[i - 1]!r} -> {items[i]!r} at {i}"})
    return out


def detect_runs(labels) -> list[dict]:
    """Run-length summary: ``[{value, start, end (inclusive), length}]``."""
    items = list(labels)
    if not items:
        return []
    runs: list[dict] = []
    cur, start = items[0], 0
    for i in range(1, len(items) + 1):
        if i == len(items) or items[i] != cur:
            runs.append({"value": cur, "start": start, "end": i - 1,
                         "length": i - start})
            if i < len(items):
                cur, start = items[i], i
    return runs
