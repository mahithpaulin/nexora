"""Closed-form numeric sequence solving (stdlib only, deterministic).

While Markov prediction answers "what usually follows", this module answers
"what MUST follow given the rule": constant finite differences (linear,
quadratic, ...), constant ratios (geometric), and additive recurrence
(Fibonacci-like). All classical, non-neural, exact for ints with a
relative tolerance for floats. Missing (None/NaN) values are skipped by
position (documented); non-numeric input raises ValueError.
"""

from __future__ import annotations

import math

_REL_TOL = 1e-9


def _clean(values) -> list:
    """Numbers in order, skipping None/NaN; raise on non-numeric.

    Integral floats (2.0) are normalized to int (2): for rule detection
    a counted 2 and a measured 2.0 obey the same recurrences, and this
    keeps integer rules exact (confidence 1.0, int outputs).
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError("values must be a list or tuple")
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"non-numeric value {v!r}")
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out.append(v)
    return out


def _is_int_like(xs) -> bool:
    return all(isinstance(x, int) and not isinstance(x, bool) for x in xs)


def _close(a, b, tol=_REL_TOL) -> bool:
    """Exact for ints; relative tolerance for floats."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    fa, fb = float(a), float(b)
    if fa == fb:
        return True
    scale = max(1.0, abs(fa), abs(fb))
    return abs(fa - fb) <= tol * scale


def _constant(xs) -> bool:
    return all(_close(x, xs[0]) for x in xs[1:])


def finite_differences(values, max_order=3):
    """Successive difference table up to max_order.

    Returns {"table": [level0, level1, ...], "constant_order": int|None}
    where constant_order is the smallest order >= 1 whose level is
    constant (None if none). Needs >= 2 points for order 1, etc.
    """
    if not isinstance(max_order, int) or max_order < 1:
        raise ValueError("max_order must be an int >= 1")
    xs = _clean(values)
    table = [list(xs)]
    for _ in range(max_order):
        prev = table[-1]
        if len(prev) < 2:
            break
        table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    const = None
    for order in range(1, len(table)):
        if len(table[order]) >= 1 and _constant(table[order]):
            const = order
            break
    return {"table": table, "constant_order": const}


def extrapolate_poly(values, order, steps=1):
    """Extend a series assuming constant order-`order` differences.

    Appends by the textbook method: extend the constant level, then
    integrate back up. Returns the new tail values (ints stay ints).
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be an int >= 1")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an int >= 1")
    xs = _clean(values)
    if len(xs) < order + 1:
        raise ValueError(f"need >= {order + 1} points for order {order}")
    fd = finite_differences(xs, max_order=order)
    if fd["constant_order"] != order:
        raise ValueError(f"order-{order} differences are not constant")
    int_like = _is_int_like(xs)
    levels = [list(lvl) for lvl in fd["table"][: order + 1]]
    out = []
    for _ in range(steps):
        levels[order].append(levels[order][-1])
        for lv in range(order - 1, -1, -1):
            levels[lv].append(levels[lv][-1] + levels[lv + 1][-1])
        v = levels[0][-1]
        out.append(int(v) if int_like and float(v).is_integer() else v)
    return out


def detect_geometric(values):
    """Constant-ratio check. Returns {"ratio", "next"} or None."""
    xs = _clean(values)
    if len(xs) < 3 or any(float(x) == 0.0 for x in xs[:-1]):
        return None
    ratios = [xs[i + 1] / xs[i] for i in range(len(xs) - 1)]
    if not _constant(ratios):
        return None
    r = ratios[0]
    nxt = xs[-1] * r
    if _is_int_like(xs) and float(nxt).is_integer():
        nxt = int(nxt)
    return {"ratio": r, "next": nxt}


def detect_additive_recurrence(values):
    """Fibonacci-like check x[n] == x[n-1] + x[n-2]. Returns {"next"} or None."""
    xs = _clean(values)
    if len(xs) < 4:
        return None
    for i in range(2, len(xs)):
        if not _close(xs[i], xs[i - 1] + xs[i - 2]):
            return None
    nxt = xs[-1] + xs[-2]
    if _is_int_like(xs) and float(nxt).is_integer():
        nxt = int(nxt)
    return {"next": nxt}


_KIND_NAMES = {1: "arithmetic", 2: "quadratic", 3: "cubic"}


def analyze_numeric_sequence(values, max_order=3, steps=1):
    """Solve a numeric sequence by rule, in order: additive recurrence,
    geometric, then polynomial (lowest constant-difference order wins).

    Returns {"kind", "params", "next" (list of `steps` values),
    "confidence" (0..1), "explanation"}. kind is arithmetic/quadratic/
    cubic/geometric/fibonacci_like/unknown. Confidence 1.0 for exact
    integer rules, 0.95 for float-tolerance matches, 0.0 for unknown.
    Never raises on numeric input (unknown when nothing fits).
    """
    xs = _clean(values)
    if len(xs) < 3:
        return {"kind": "unknown", "params": {}, "next": [], "confidence": 0.0,
                "explanation": f"need >= 3 points for rule detection, got {len(xs)}."}
    exact = _is_int_like(xs)
    conf = 1.0 if exact else 0.95
    add = detect_additive_recurrence(xs)
    if add is not None:
        tail = [add["next"]]
        for _ in range(steps - 1):
            tail.append(tail[-1] + (tail[-2] if len(tail) > 1 else xs[-1]))
        return {"kind": "fibonacci_like", "params": {"rule": "x[n]=x[n-1]+x[n-2]"},
                "next": tail[:steps], "confidence": conf,
                "explanation": f"each term is the sum of the two before it "
                               f"({xs[-2]}+{xs[-1]}={add['next']}); confidence {conf:.2f}."}
    geo = detect_geometric(xs)
    if geo is not None:
        tail, v = [], xs[-1]
        for _ in range(steps):
            v = v * geo["ratio"]
            tail.append(int(v) if exact and float(v).is_integer() else v)
        return {"kind": "geometric", "params": {"ratio": geo["ratio"]},
                "next": tail, "confidence": conf,
                "explanation": f"constant ratio {geo['ratio']} between consecutive terms; "
                               f"next {tail[0]}; confidence {conf:.2f}."}
    fd = finite_differences(xs, max_order=max_order)
    order = fd["constant_order"]
    # Degenerate fits rejected: the constant level must hold >= 2 values
    # (any N points are trivially interpolated by an order-(N-1) polynomial,
    # which predicts nothing). Needs >= order+2 points.
    if order is not None and len(fd["table"][order]) < 2:
        order = None
    if order is not None:
        tail = extrapolate_poly(xs, order, steps)
        kind = _KIND_NAMES.get(order, f"poly-order-{order}")
        d = fd["table"][order][0]
        return {"kind": kind, "params": {"order": order, "difference": d},
                "next": tail, "confidence": conf,
                "explanation": f"order-{order} differences are constant ({d}); "
                               f"next {tail[0]}; confidence {conf:.2f}."}
    return {"kind": "unknown", "params": {}, "next": [], "confidence": 0.0,
            "explanation": "no constant-difference, constant-ratio, or additive-recurrence rule fits."}
