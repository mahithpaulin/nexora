"""Statistical feature extraction for a numeric series (stdlib only).

``describe_series(values)`` summarises the valid (non-missing) entries.
Missing values (``None``/``NaN``) are skipped explicitly and counted under
``"missing"``; any other non-numeric entry raises ``ValueError``.

Returned dict keys: ``n`` (valid count), ``missing``, ``mean``, ``median``,
``variance`` (population), ``stdev`` (population), ``min``, ``max``,
``p25``/``p50``/``p75`` plus a nested ``percentiles`` dict, ``iqr``
(``p75-p25``), ``cv`` (``stdev/|mean|``; ``0.0`` for constant series,
``None`` when undefined), ``zscores`` (list parallel to the input: ``None``
where input was missing, ``0.0`` everywhere when ``stdev == 0``).

Empty / all-missing input is safe: numeric stats are ``None``, ``n == 0``.
Constant input is safe: variance/stdev ``0.0``, zscores all ``0.0``.
"""

from __future__ import annotations

import math
import statistics


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100) of sorted data."""
    k = len(sorted_vals)
    if k == 1:
        return sorted_vals[0]
    pos = (p / 100.0) * (k - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def describe_series(values: list[float]) -> dict:
    """Describe a numeric series; see module docstring for keys/meaning."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("values must be a list or tuple")
    clean: list[float] = []
    missing = 0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            missing += 1
            continue
        if not isinstance(v, (int, float)):
            raise ValueError(f"non-numeric value {v!r} (only None/NaN are skipped)")
        clean.append(float(v))
    n = len(clean)
    zscores: list[float | None] = []
    if n == 0:
        return {
            "n": 0, "missing": missing,
            "mean": None, "median": None, "variance": None, "stdev": None,
            "min": None, "max": None,
            "p25": None, "p50": None, "p75": None,
            "percentiles": {"p25": None, "p50": None, "p75": None},
            "iqr": None, "cv": None,
            "zscores": [None] * len(list(values)),
        }
    mean = statistics.fmean(clean)
    median = statistics.median(clean)
    var = statistics.pvariance(clean)
    sd = statistics.pstdev(clean)
    srt = sorted(clean)
    p25 = _percentile(srt, 25)
    p50 = _percentile(srt, 50)
    p75 = _percentile(srt, 75)
    iqr = p75 - p25
    if mean != 0.0:
        cv: float | None = sd / abs(mean)
    else:
        cv = 0.0 if sd == 0.0 else None
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            zscores.append(None)
        elif sd == 0.0:
            zscores.append(0.0)
        else:
            zscores.append((float(v) - mean) / sd)
    return {
        "n": n, "missing": missing,
        "mean": mean, "median": median, "variance": var, "stdev": sd,
        "min": srt[0], "max": srt[-1],
        "p25": p25, "p50": p50, "p75": p75,
        "percentiles": {"p25": p25, "p50": p50, "p75": p75},
        "iqr": iqr, "cv": cv,
        "zscores": zscores,
    }
