"""Pairwise correlation utilities (stdlib only, deterministic)."""
import math


def _clean_number(x):
    """Return float(x) or None if missing; raise ValueError if non-numeric."""
    if x is None:
        return None
    if isinstance(x, bool):
        raise ValueError(f"non-numeric entry: {x!r}")
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    raise ValueError(f"non-numeric entry: {x!r}")


def _clean_columns(columns):
    """Validate mapping -> (names, cleaned dict of float|None lists)."""
    names = list(columns.keys())
    cleaned = {}
    for name in names:
        vals = columns[name]
        if not isinstance(vals, (list, tuple)):
            raise ValueError(f"column {name!r} must be a list/tuple")
        cleaned[name] = [_clean_number(v) for v in vals]
    return names, cleaned


def _pearson_of_pairs(pairs):
    """Pearson r for list of (x, y) floats; 0.0 if n==0 or zero variance."""
    n = len(pairs)
    if n == 0:
        return 0.0
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs) / n
    vx = sum((p[0] - mx) ** 2 for p in pairs) / n
    vy = sum((p[1] - my) ** 2 for p in pairs) / n
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    r = cov / (math.sqrt(vx) * math.sqrt(vy))
    return max(-1.0, min(1.0, r))


def covariance_matrix(columns: dict):
    """Population covariance with pairwise-complete deletion.

    Input: columns dict name -> list of numbers/None/NaN. Output:
    {"names", "matrix" (n x n, diagonal = variances), "n" ({(a,b): pair
    count})}. Covariances are unbounded floats. Missing pairs skipped;
    differing lengths pair by index up to min length. Non-numeric entries
    raise ValueError. n==0 -> 0.0.
    """
    names, cleaned = _clean_columns(columns)
    k = len(names)
    matrix = [[0.0] * k for _ in range(k)]
    n = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ca, cb = cleaned[a], cleaned[b]
            m = min(len(ca), len(cb))
            pairs = [(ca[t], cb[t]) for t in range(m)
                     if ca[t] is not None and cb[t] is not None]
            c = len(pairs)
            n[(a, b)] = c
            if c == 0:
                matrix[i][j] = 0.0
            else:
                mx = sum(p[0] for p in pairs) / c
                my = sum(p[1] for p in pairs) / c
                matrix[i][j] = sum((p[0] - mx) * (p[1] - my)
                                   for p in pairs) / c
    return {"names": names, "matrix": matrix, "n": n}


def correlation_matrix(columns: dict):
    """Pearson correlation with pairwise-complete deletion.

    Output: {"names", "matrix" (r in -1..1; 0.0 where undefined —
    empty pair or zero variance), "n"}. Diagonal is 1.0 iff the column
    has variance > 0, else 0.0. Non-numeric entries raise ValueError.
    """
    names, cleaned = _clean_columns(columns)
    k = len(names)
    matrix = [[0.0] * k for _ in range(k)]
    n = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ca, cb = cleaned[a], cleaned[b]
            m = min(len(ca), len(cb))
            pairs = [(ca[t], cb[t]) for t in range(m)
                     if ca[t] is not None and cb[t] is not None]
            n[(a, b)] = len(pairs)
            if i == j:
                if len(pairs) == 0:
                    matrix[i][j] = 0.0
                else:
                    mx = sum(p[0] for p in pairs) / len(pairs)
                    vx = sum((p[0] - mx) ** 2 for p in pairs) / len(pairs)
                    matrix[i][j] = 1.0 if vx > 0.0 else 0.0
            else:
                matrix[i][j] = _pearson_of_pairs(pairs)
    return {"names": names, "matrix": matrix, "n": n}


def cross_correlation(a: list, b: list, max_lag: int = 10):
    """Lagged Pearson r between two series.

    Output {lag: r} for lag in -max_lag..max_lag, r in -1..1. Lag k
    correlates a[i] with b[i+k]: k>0 means b leads (b shifted forward),
    k<0 means a leads. Pairwise-complete per lag; 0.0 where undefined.
    Raises ValueError on non-numeric entries or negative max_lag.
    """
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) \
            or max_lag < 0:
        raise ValueError("max_lag must be a non-negative int")
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        raise ValueError("a and b must be lists/tuples")
    ca = [_clean_number(v) for v in a]
    cb = [_clean_number(v) for v in b]
    out = {}
    for k in range(-max_lag, max_lag + 1):
        pairs = []
        for i in range(len(ca)):
            j = i + k
            if 0 <= j < len(cb) and ca[i] is not None \
                    and cb[j] is not None:
                pairs.append((ca[i], cb[j]))
        out[k] = _pearson_of_pairs(pairs)
    return out


def find_strong_correlations(corr_result: dict, threshold: float = 0.7):
    """Filter a correlation matrix for strong pairs (|r| >= threshold).

    Output [{"a", "b", "r" (-1..1), "strength" (0..1 = |r|), "reason"}]
    sorted by -strength (ties by a, b). Upper-triangle pairs only.
    """
    names = corr_result.get("names", [])
    matrix = corr_result.get("matrix", [])
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(matrix[i][j])
            if abs(r) >= threshold:
                out.append({"a": names[i], "b": names[j], "r": r,
                            "strength": abs(r),
                            "reason": f"|r|={abs(r):.3f} >= {threshold}"})
    out.sort(key=lambda d: (-d["strength"], d["a"], d["b"]))
    return out


def _lookup_pair_count(n: dict, a, b):
    """Best-effort pair-count lookup supporting tuple or 'a|b' keys."""
    for key in ((a, b), (b, a), f"{a}|{b}", f"{b}|{a}",
                f"{a},{b}", f"{b},{a}"):
        if key in n:
            return int(n[key])
    return 0


def find_correlation_patterns(columns: dict, threshold: float = 0.7):
    """Mine strong pairwise correlations as pattern dicts (type "correlation").

    Features {"a", "b", "r" (-1..1), "strength" (0..1), "n"}; sequence
    [a, b]; frequency = pair count n; confidence = |r|. Sorted by
    -strength. Empty list if none pass the threshold.
    """
    corr = correlation_matrix(columns)
    strong = find_strong_correlations(corr, threshold=threshold)
    patterns = []
    for s in strong:
        cnt = _lookup_pair_count(corr["n"], s["a"], s["b"])
        patterns.append({"id": None, "type": "correlation",
                         "features": {"a": s["a"], "b": s["b"],
                                      "r": s["r"],
                                      "strength": s["strength"], "n": cnt},
                         "sequence": [s["a"], s["b"]],
                         "frequency": cnt, "occurrences": [],
                         "confidence": abs(s["r"])})
    return patterns
