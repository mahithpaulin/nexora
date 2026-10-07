"""Robust anomaly detection: median/MAD rolling scores, level shifts, severity.

Stdlib only, deterministic. Complements (does not replace) the global
mean/std detector: global z-scores stay for batch outliers with cited
z-values, while this module scores each point against its own recent
past (causal rolling window) using statistics that a single spike
cannot distort.

- ``modified_z``: 0.6745*(x - median)/MAD, the standard robust z.
- ``robust_detect``: causal rolling median/MAD scoring (kind "robust"),
  optionally on seasonality-centered residuals (``period``).
- ``level_shift_records``: mean-shift change points as anomalies
  (kind "level_shift") via ``discovery.change_points``.
- ``severity_of``: score -> low/medium/high/critical.
"""

from __future__ import annotations

import math
import statistics

try:
    from nexora.discovery.change_points import change_points as _change_points
except ImportError:  # pragma: no cover - package always ships it
    _change_points = None


def severity_of(score) -> str:
    """Map a 0..1 anomaly score to a severity level (deterministic)."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "low"
    if s != s:  # NaN
        return "low"
    if s < 0.5:
        return "low"
    if s < 0.7:
        return "medium"
    if s < 0.9:
        return "high"
    return "critical"


def median_of(xs) -> float | None:
    try:
        vals = [float(x) for x in xs]
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    return float(statistics.median(vals))


def mad_of(xs, med=None) -> float | None:
    try:
        vals = [float(x) for x in xs]
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    if med is None:
        med = statistics.median(vals)
    return float(statistics.median([abs(v - med) for v in vals]))


def modified_z(x, med, mad) -> float:
    """Robust z-score; MAD == 0 means the past was perfectly flat."""
    try:
        xf, mf = float(x), float(med)
    except (TypeError, ValueError):
        return 0.0
    if mad is None:
        return 0.0
    try:
        madf = float(mad)
    except (TypeError, ValueError):
        return 0.0
    if madf == 0.0:
        return 0.0 if xf == mf else math.inf
    return 0.6745 * (xf - mf) / madf


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _rows(values_or_rows):
    out = []
    seq = list(values_or_rows) if values_or_rows is not None else []
    for i, item in enumerate(seq):
        if isinstance(item, dict):
            idx = item.get("index", i)
            val = item.get("value", item.get("label"))
            out.append({"index": idx, "value": val})
        else:
            out.append({"index": i, "value": item})
    return out


def robust_detect(values_or_rows, window=20, warmup=None, threshold=3.5,
                  period=None):
    """Causal rolling median/MAD anomaly scores (kind "robust").

    Each numeric point is scored against the ``window`` points before
    it (never its future, never itself); scoring starts once at least
    3 history points exist. ``warmup`` overrides that minimum.
    ``period`` centers each point on its seasonal phase median first
    (residual scoring), so a regular cycle does not mask real spikes.
    Returns ``[{index, value, z, score, kind, causes, explanation,
    severity}]`` with ``z`` = the modified z (``inf`` on jumps from a
    flat past, ``None`` never — a number was always measured here).
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive int, got %r" % (window,))
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        thr = 3.5
    if not thr > 0:
        raise ValueError("threshold must be positive, got %r" % (threshold,))
    rows = _rows(values_or_rows)
    nums = [(r["index"], float(r["value"])) for r in rows if _is_num(r["value"])]
    need = warmup if isinstance(warmup, int) and warmup >= 1 else 3
    per = None
    if isinstance(period, int) and period >= 2 and len(nums) >= 2 * period:
        try:
            groups: dict[int, list[float]] = {}
            for k, (_, v) in enumerate(nums):
                groups.setdefault(k % period, []).append(v)
            per = {ph: float(statistics.median(g)) for ph, g in groups.items()}
        except (ValueError, statistics.StatisticsError):
            per = None
    out = []
    hist: list[float] = []
    for k, (idx, val) in enumerate(nums):
        x = val - per[k % period] if per is not None else val
        if len(hist) >= need:
            base = hist[-window:]
            med = statistics.median(base)
            mad = statistics.median([abs(v - med) for v in base])
            rz = modified_z(x, med, mad)
            if abs(rz) >= thr or rz in (math.inf, -math.inf):
                score = 1.0 if rz in (math.inf, -math.inf) else min(1.0, abs(rz) / (thr * 2.0))
                tail = "past-%d" % len(base)
                if per is not None:
                    tail += " deseasonalized(period=%d)" % period
                out.append({
                    "index": idx, "value": val, "z": rz, "score": score,
                    "kind": "robust",
                    "causes": ["robust z-score %.2f vs %s median %.3f (MAD %.3f, threshold %.2f)"
                               % (rz, tail, med, mad, thr)],
                    "explanation": ("Robust outlier at index %s: value %s deviates "
                                    "robust-z=%.2f from trailing median %.3f (MAD %.3f, "
                                    "threshold %.2f), score %.2f."
                                    % (idx, val, rz, med, mad, thr, score)),
                    "severity": severity_of(score),
                })
        hist.append(x if per is not None else val)
    out.sort(key=lambda d: (d["index"], d["kind"]))
    return out


def level_shift_records(values_or_rows, window=10, threshold_z=3.0):
    """Mean-shift change points as anomaly records (kind "level_shift").

    Thin wrapper over ``discovery.change_points``: positions map back
    to row indices, ``z`` is None (a windowed mean shift, not a point
    z-score), score scales with the shift's |z| (inf -> 1.0).
    """
    if _change_points is None:
        return []
    rows = _rows(values_or_rows)
    nums = [(r["index"], float(r["value"])) for r in rows if _is_num(r["value"])]
    if len(nums) < 2 * window:
        return []
    try:
        cps = _change_points([v for _, v in nums], window=window, threshold_z=threshold_z) or []
    except (ValueError, TypeError):
        return []
    out = []
    for cp in cps:
        try:
            s = int(cp.get("split", cp.get("index", -1)))
            z = float(cp.get("z", 0.0))
            bm = float(cp.get("before_mean", 0.0))
            am = float(cp.get("after_mean", 0.0))
        except (TypeError, ValueError):
            continue
        if not 0 <= s < len(nums):
            continue
        idx, val = nums[s]
        score = 1.0 if z in (math.inf, -math.inf) else min(1.0, abs(z) / (threshold_z * 2.0))
        out.append({
            "index": idx, "value": val, "z": None, "score": score,
            "kind": "level_shift",
            "causes": ["windowed mean shifted %.3f -> %.3f (|z|=%.2f, threshold %.2f)"
                       % (bm, am, z, threshold_z)],
            "explanation": ("Level shift at index %s (value %s): trailing mean %.3f "
                            "shifted to %.3f (|z|=%.2f, threshold %.2f), score %.2f."
                            % (idx, val, bm, am, z, threshold_z, score)),
            "severity": severity_of(score),
        })
    out.sort(key=lambda d: (d["index"], d["kind"]))
    return out
