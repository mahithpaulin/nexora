"""Dynamic Time Warping distance with Sakoe-Chiba band (stdlib only).

``dtw_distance(a, b, window=None, scale=1.0)`` aligns two numeric series
allowing local time shifts; local cost is ``abs(x - y)``. Returns
``(dist, path, similarity)`` where ``path`` is the optimal warping path
as ``[(i, j), ...]`` from ``(0, 0)`` to ``(n-1, m-1)``, and
``similarity = 1 / (1 + dist / scale)`` in ``0..1`` (``1.0`` = identical).

Complexity is ``O(n*m)`` time and ``O(n*m)`` space (full cost matrix is
kept for path backtracking); ``window=w`` (Sakoe-Chiba half-band,
``|i-j| <= w``) caps work to ``O(n*w)``. ``window < abs(n-m)`` raises
``ValueError`` (target unreachable). Empty inputs, ``None``/``NaN``
entries, or ``scale <= 0`` raise ``ValueError``. Tie-breaking prefers the
diagonal step, so results are deterministic.
"""

from __future__ import annotations


def _numvec(seq, name: str) -> list[float]:
    if not isinstance(seq, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if len(seq) == 0:
        raise ValueError(f"{name} must be non-empty")
    out: list[float] = []
    for v in seq:
        if v is None or not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must hold only real numbers "
                             f"(None/NaN not allowed), got {v!r}")
        out.append(float(v))
    return out


def dtw_distance(a, b, window=None, scale=1.0):
    """DTW alignment; returns ``(dist, path, similarity)`` (see docstring)."""
    xa, xb = _numvec(a, "a"), _numvec(b, "b")
    n, m = len(xa), len(xb)
    if not isinstance(scale, (int, float)) or scale != scale or scale <= 0:
        raise ValueError(f"scale must be a positive number, got {scale!r}")
    if window is None:
        w = max(n, m)
    else:
        if not isinstance(window, int) or window < 0:
            raise ValueError("window must be a non-negative int or None")
        w = window
    if w < abs(n - m):
        raise ValueError(f"window {w} < |n-m|={abs(n - m)}: end unreachable")
    inf = float("inf")
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - w)
        j1 = min(m, i + w)
        for j in range(j0, j1 + 1):
            cost = abs(xa[i - 1] - xb[j - 1])
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    dist = D[n][m]
    path: list[tuple[int, int]] = [(n - 1, m - 1)]
    i, j = n, m
    while i > 1 or j > 1:
        opts = []
        if i > 1 and j > 1:
            opts.append((D[i - 1][j - 1], i - 1, j - 1))  # diagonal first
        if i > 1:
            opts.append((D[i - 1][j], i - 1, j))
        if j > 1:
            opts.append((D[i][j - 1], i, j - 1))
        _, i, j = min(opts, key=lambda t: t[0])
        path.append((i - 1, j - 1))
    path.reverse()
    similarity = 1.0 / (1.0 + dist / float(scale))
    return dist, path, similarity
