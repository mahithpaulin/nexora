"""Temporal features for ordered numeric series (stdlib only).

Missing values (``None``/``NaN``) are skipped explicitly (order preserved);
other non-numeric entries raise ``ValueError``.

- ``moving_average``: trailing-window means; ``None`` where a window holds
  no valid value. Output length == input length.
- ``autocorrelation``: ``{lag: r}`` with ``r`` in ``-1..1`` (global-mean
  estimator); ``0.0`` when variance is zero.
- ``detect_trend``: least-squares ``slope`` on ``x = 0..n-1``,
  ``direction`` in ``rising|falling|flat``, ``strength``/``r2`` (R^2,
  ``0..1``) plus a ``reason`` string.
- ``fft_periodicity``: naive O(n^2) DFT. Input is downsampled to at most
  256 points (even stride) to bound cost; periods are reported in original
  sample units. Returns ``{"periods", "strengths" (0..1, best=1.0),
  "method": "dft-naive"}``.
"""

from __future__ import annotations

import math


def _clean(values) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("values must be a list or tuple")
    out: list[float] = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if not isinstance(v, (int, float)):
            raise ValueError(f"non-numeric value {v!r}")
        out.append(float(v))
    return out


def moving_average(values, window=3) -> list[float | None]:
    """Trailing moving average; ``None`` where the window is all-missing."""
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive int")
    vals = list(values)
    out: list[float | None] = []
    for i in range(len(vals)):
        seg = [float(v) for v in vals[max(0, i - window + 1): i + 1]
               if v is not None and not (isinstance(v, float) and math.isnan(v))]
        out.append(sum(seg) / len(seg) if seg else None)
    return out


def autocorrelation(values, max_lag=10) -> dict[int, float]:
    """Autocorrelation ``{lag: r}`` for lags ``1..max_lag`` (capped at n-1)."""
    if not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive int")
    x = _clean(values)
    n = len(x)
    if n < 2:
        return {}
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x)
    top = min(max_lag, n - 1)
    if var == 0.0:
        return {lag: 0.0 for lag in range(1, top + 1)}
    out: dict[int, float] = {}
    for lag in range(1, top + 1):
        cov = sum((x[i] - mean) * (x[i + lag] - mean) for i in range(n - lag))
        r = cov / var
        out[lag] = max(-1.0, min(1.0, r))
    return out


def detect_trend(values) -> dict:
    """Least-squares trend: slope, direction, R^2 strength, reason."""
    x = _clean(values)
    n = len(x)
    if n < 2:
        return {"slope": 0.0, "direction": "flat", "strength": 0.0,
                "r2": 0.0, "reason": "need >= 2 valid samples"}
    mean_x = (n - 1) / 2.0
    mean_y = sum(x) / n
    sxx = sum((i - mean_x) ** 2 for i in range(n))
    sxy = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(x))
    slope = sxy / sxx if sxx else 0.0
    sst = sum((y - mean_y) ** 2 for y in x)
    ssr = sum((y - (mean_y + slope * (i - mean_x))) ** 2 for i, y in enumerate(x))
    r2 = 0.0 if sst == 0.0 else max(0.0, min(1.0, 1.0 - ssr / sst))
    if abs(slope) <= 1e-9 or sst == 0.0:
        direction = "flat"
    elif slope > 0:
        direction = "rising"
    else:
        direction = "falling"
    return {"slope": slope, "direction": direction, "strength": r2,
            "r2": r2, "reason": f"slope={slope:.4g} via OLS, R^2={r2:.4f}"}


def fft_periodicity(values, top_k=5) -> dict:
    """Naive-DFT periodicity; downsamples to <= 256 points (see docstring)."""
    x = _clean(values)
    m = len(x)
    if m < 4:
        return {"periods": [], "strengths": [], "method": "dft-naive",
                "reason": "need >= 4 valid samples"}
    stride = max(1, math.ceil(m / 256))
    xs = x[::stride]
    m2 = len(xs)
    m0 = sum(xs) / m2
    xs = [v - m0 for v in xs]  # remove DC component
    specs: list[tuple[float, float]] = []
    for k in range(1, m2 // 2 + 1):
        re = sum(v * math.cos(2 * math.pi * k * j / m2) for j, v in enumerate(xs))
        im = -sum(v * math.sin(2 * math.pi * k * j / m2) for j, v in enumerate(xs))
        mag = math.hypot(re, im) / m2
        specs.append(((m2 / k) * stride, mag))
    peak = max(mag for _, mag in specs)
    if peak == 0.0:
        return {"periods": [], "strengths": [], "method": "dft-naive",
                "reason": "constant series: no spectral peaks"}
    ranked = sorted(specs, key=lambda t: (-t[1], t[0]))[: max(1, top_k)]
    return {"periods": [round(p, 6) for p, _ in ranked],
            "strengths": [round(mag / peak, 6) for _, mag in ranked],
            "method": "dft-naive",
            "reason": f"naive DFT on {m2} pts (stride {stride}); top-{len(ranked)} peaks"}
