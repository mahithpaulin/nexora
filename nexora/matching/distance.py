"""Numeric vector distances (stdlib only).

``euclidean`` / ``manhattan`` return raw distances (``>= 0``; ``0`` =
identical). Inputs must be non-empty equal-length lists/tuples of real
numbers; ``None``/``NaN`` entries, length mismatch, or empty inputs raise
``ValueError`` (missing data is never silently ignored).

``normalized_similarity(dist, scale)`` maps any distance to ``0..1`` via
``1 / (1 + dist / scale)``: ``1.0`` = identical, ``-> 0`` as distance
grows. Use it to turn a raw distance into the required similarity score.
"""

from __future__ import annotations

import math


def _nums(seq, name: str) -> list[float]:
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


def euclidean(a, b) -> float:
    """Euclidean (L2) distance; ``0.0`` for identical vectors."""
    xa, xb = _nums(a, "a"), _nums(b, "b")
    if len(xa) != len(xb):
        raise ValueError(f"length mismatch: {len(xa)} != {len(xb)}")
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(xa, xb)))


def manhattan(a, b) -> float:
    """Manhattan (L1) distance; ``0.0`` for identical vectors."""
    xa, xb = _nums(a, "a"), _nums(b, "b")
    if len(xa) != len(xb):
        raise ValueError(f"length mismatch: {len(xa)} != {len(xb)}")
    return sum(abs(u - v) for u, v in zip(xa, xb))


def normalized_similarity(dist: float, scale: float = 1.0) -> float:
    """Map a distance to similarity ``0..1`` via ``1/(1+dist/scale)``."""
    if dist is None or not isinstance(dist, (int, float)) or dist != dist:
        raise ValueError(f"dist must be a real number, got {dist!r}")
    if not isinstance(scale, (int, float)) or scale != scale or scale <= 0:
        raise ValueError(f"scale must be a positive number, got {scale!r}")
    if dist < 0:
        raise ValueError(f"dist must be >= 0, got {dist!r}")
    return 1.0 / (1.0 + float(dist) / float(scale))
