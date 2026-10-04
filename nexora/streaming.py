"""Streaming (online) statistics and causal anomaly flags for Nexora.

Memory: RunningStats is O(1); SlidingStats is O(window); the stream
helpers hold O(window) state only and never buffer the full series.
Score ranges: z is unbounded (later +/-inf for zero-variance jumps);
anomaly ``score`` is in [0.5, 1.0].

Behavior changes (v2):
- Validity is ``_ok``: only finite int/float excluding bool are valid
  samples. None/NaN/str/bool/inf/-inf and any other non-numeric or
  non-finite value is treated like missing (counted in ``missing``,
  excluded from sums, never raises). Previously strings raised via
  float() and bool/inf were folded into the statistics.
- ``stream_anomalies`` flags constant-history jumps: past stdev == 0
  with past n >= 2 and a valid current value != mean yields z=+/-inf,
  score=1.0 and a "zero-variance baseline jump" explanation.
- Complexities unchanged: RunningStats O(1), SlidingStats O(1) per
  update/query with O(window) space, stream helpers O(n) time with
  O(window) space.
"""

from __future__ import annotations

import math
from collections import deque

__all__ = ["RunningStats", "SlidingStats", "stream_anomalies", "streaming_describe"]


def _is_missing(x) -> bool:
    """Return True for None or float NaN (legacy missing check).

    Kept for backward compatibility; the canonical validity gate is
    now ``_ok`` (finite int/float, non-bool). Anything not ``_ok`` is
    treated like missing by all update/stream paths.
    """
    if x is None:
        return True
    return isinstance(x, float) and math.isnan(x)


def _ok(x) -> bool:
    """True for valid numeric samples: int/float, non-bool, finite.

    Excludes None, NaN, +/-inf, bool, str and any other type. O(1).
    """
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return math.isfinite(x)
    return False


class RunningStats:
    """Welford online mean/variance over an unbounded stream.

    Formulas (Welford, population variance):
        n=0, mean=0, M2=0
        on x: n+=1; d=x-mean; mean+=d/n; d2=x-mean; M2+=d*d2
        variance = M2/n; stdev = sqrt(variance)

    Merge (Chan parallel combine) for a,b with n,mean,M2:
        n=n_a+n_b; d=mean_b-mean_a
        mean=mean_a+d*n_b/n; M2=M2_a+M2_b+d*d*n_a*n_b/n

    Time O(1) per update/merge; space O(1).
    mean/variance/stdev are None when n==0 (no valid points seen).
    variance is the population variance (>= 0); stdev >= 0.
    ``missing`` counts skipped values: None/NaN plus (v2) any
    non-numeric (e.g. str), bool, or non-finite (inf/-inf) input,
    per ``_ok``. Such values never raise and never bias the sums.
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._missing: int = 0

    def update(self, x) -> "RunningStats":
        """Fold one value in; skip invalid (count in ``missing``). O(1)."""
        if not _ok(x):
            self._missing += 1
            return self
        xf = float(x)
        self._n += 1
        delta = xf - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (xf - self._mean)
        return self

    def merge(self, other: "RunningStats") -> "RunningStats":
        """In-place parallel combine with ``other``. O(1) time/space."""
        if not isinstance(other, RunningStats):
            raise TypeError("merge expects a RunningStats")
        if other._n == 0:
            self._missing += other._missing
            return self
        if self._n == 0:
            self._n, self._mean, self._m2 = other._n, other._mean, other._m2
            self._missing += other._missing
            return self
        n1, n2 = self._n, other._n
        delta = other._mean - self._mean
        total = n1 + n2
        self._mean = self._mean + delta * n2 / total
        self._m2 = self._m2 + other._m2 + delta * delta * n1 * n2 / total
        self._n = total
        self._missing += other._missing
        return self

    @property
    def n(self) -> int:
        """Valid (non-missing) points seen."""
        return self._n

    @property
    def missing(self) -> int:
        """Count of skipped values (None/NaN plus v2: non-numeric/bool/non-finite)."""
        return self._missing

    @property
    def mean(self) -> float | None:
        """Running mean, or None if n==0. Unbounded float."""
        return self._mean if self._n else None

    @property
    def variance(self) -> float | None:
        """Population variance M2/n (>= 0), or None if n==0."""
        if not self._n:
            return None
        v = self._m2 / self._n
        return v if v >= 0.0 else 0.0

    @property
    def stdev(self) -> float | None:
        """Population stdev (>= 0), or None if n==0."""
        v = self.variance
        return math.sqrt(v) if v is not None else None

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"RunningStats(n={self._n}, mean={self.mean}, stdev={self.stdev}, missing={self._missing})"


class SlidingStats:
    """Fixed-window stats; missing/invalid values occupy a slot as None.

    update(None/NaN) — and (v2) any non-``_ok`` value such as str, bool,
    or inf — appends None to the deque (advancing the window position
    and evicting the oldest entry) but contributes nothing to sums;
    mean/variance/stdev cover valid entries only. This keeps alignment
    with the input index (one slot per observation) while not letting
    gaps or junk bias the sums.

    Time O(1) per update/query; space O(window).
    mean/variance/stdev are None when the window holds no valid values.
    variance is population variance (>= 0).
    """

    def __init__(self, window: int) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive int")
        self._window = window
        self._buf: deque = deque(maxlen=window)
        self._sum: float = 0.0
        self._sumsq: float = 0.0
        self._valid: int = 0
        self._missing: int = 0

    def update(self, x) -> "SlidingStats":
        """Append x (or a None placeholder for invalid), evicting oldest. O(1)."""
        if not _ok(x):
            if len(self._buf) == self._window:
                evicted = self._buf[0]
                if evicted is not None:
                    self._sum -= evicted
                    self._sumsq -= evicted * evicted
                    self._valid -= 1
            self._buf.append(None)
            self._missing += 1
            return self
        xf = float(x)
        if len(self._buf) == self._window:
            evicted = self._buf[0]
            if evicted is not None:
                self._sum -= evicted
                self._sumsq -= evicted * evicted
                self._valid -= 1
        self._buf.append(xf)
        self._sum += xf
        self._sumsq += xf * xf
        self._valid += 1
        return self

    @property
    def window(self) -> int:
        """Configured window length."""
        return self._window

    @property
    def n(self) -> int:
        """Valid (non-missing) values currently in the window."""
        return self._valid

    @property
    def missing(self) -> int:
        """Total invalid values seen (cumulative, not windowed)."""
        return self._missing

    @property
    def mean(self) -> float | None:
        """Window mean, or None if no valid values. O(1)."""
        if not self._valid:
            return None
        return self._sum / self._valid

    @property
    def variance(self) -> float | None:
        """Population variance over valid window values, else None. O(1)."""
        if not self._valid:
            return None
        m = self._sum / self._valid
        v = self._sumsq / self._valid - m * m
        return v if v > 0.0 else 0.0

    @property
    def stdev(self) -> float | None:
        """Population stdev over valid window values, else None. O(1)."""
        v = self.variance
        return math.sqrt(v) if v is not None else None

    def __len__(self) -> int:
        return self._valid

    def __repr__(self) -> str:
        return f"SlidingStats(window={self._window}, n={self._valid}, mean={self.mean})"


def stream_anomalies(values, window: int = 50, z_threshold: float = 3.0, warmup=None) -> list:
    """Causal (past-only) z-score anomaly flags over a stream.

    For each index i, stats cover values[max(0,i-window):i] — strictly
    past observations, never including values[i] and never looking
    ahead. Flag i when the value is valid and either past stdev > 0
    with |x-mean|/stdev >= z_threshold, or past stdev == 0 with past
    n >= 2 and value != mean (zero-variance baseline jump: z=+/-inf,
    score=1.0). Unlike batch detection (two-sided statistics), every
    decision uses only history, so this is deployable online.

    Only ``_ok`` values are ever scored; anything else is treated like
    missing and never raises. Time O(n); space O(window). Score =
    min(1, |z|/(2*threshold)) in [0.5, 1.0] (jumps score exactly 1.0);
    z unbounded (+/-inf for jumps). No flags before ``warmup`` points
    seen (warmup defaults to ``window``).
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive int")
    if not isinstance(z_threshold, (int, float)) or not z_threshold > 0:
        raise ValueError("z_threshold must be positive")
    if warmup is None:
        warmup = window
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative int or None")
    stats = SlidingStats(window)
    out: list = []
    for i, x in enumerate(values):
        m = stats.mean
        s = stats.stdev
        n = stats.n
        if i >= warmup and _ok(x) and m is not None and s is not None:
            xf = float(x)
            if s > 0:
                z = (xf - m) / s
                if abs(z) >= z_threshold:
                    score = abs(z) / (2.0 * float(z_threshold))
                    out.append({
                        "index": i,
                        "value": x,
                        "z": z,
                        "score": min(1.0, score),
                        "explanation": (
                            f"value {x!r} at index {i} has z={z:.2f} "
                            f"(|z| >= {float(z_threshold):.2f}) vs past-{window} "
                            f"mean={m:.4f} stdev={s:.4f}"
                        ),
                    })
            elif s == 0.0 and n >= 2 and xf != m:
                z = math.inf if xf > m else -math.inf
                out.append({
                    "index": i,
                    "value": x,
                    "z": z,
                    "score": 1.0,
                    "explanation": (
                        f"value {x!r} at index {i} is a zero-variance baseline jump "
                        f"(z={'inf' if z > 0 else '-inf'}) vs past-{window} "
                        f"mean={m:.4f} stdev=0.0000 (n={n})"
                    ),
                })
        stats.update(x)
    return out


def streaming_describe(values, window: int = 50) -> dict:
    """One-pass summary: full-stream and recent-window location/scale.

    Only ``_ok`` values feed the sums; anything else counts as missing
    and never raises. Time O(n); space O(window). Means/stdevs are
    unbounded floats (None when empty); n/missing are ints.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive int")
    full = RunningStats()
    recent = SlidingStats(window)
    for x in values:
        full.update(x)
        recent.update(x)
    return {
        "n": full.n,
        "missing": full.missing,
        "mean": full.mean,
        "stdev": full.stdev,
        "recent_mean": recent.mean,
        "recent_stdev": recent.stdev,
    }
