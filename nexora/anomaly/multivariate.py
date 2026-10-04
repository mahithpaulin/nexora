"""Multivariate anomaly detection via Mahalanobis distance (stdlib only).

Anomaly score 0..1 (higher = more anomalous): score = d2/(d2+k) with
k = dimensionality, so 0 at the mean and -> 1 as squared distance
d2 -> infinity (monotonic).
"""

import math


def _is_valid_number(v):
    """True for finite int/float (excludes None, NaN, inf, bool)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    return False


def mean_vector(vectors):
    """Per-dimension means, pairwise over valid entries.

    Dimension j averages rows with a valid numeric value at index j;
    dimensions with no valid values yield 0.0. Returns [] for empty input.
    """
    if not vectors:
        return []
    dims = 0
    for r in vectors:
        if isinstance(r, (list, tuple)):
            if len(r) > dims:
                dims = len(r)
    if dims == 0:
        return []
    sums = [0.0] * dims
    counts = [0] * dims
    for r in vectors:
        if not isinstance(r, (list, tuple)):
            continue
        for j in range(min(len(r), dims)):
            v = r[j]
            if _is_valid_number(v):
                sums[j] += float(v)
                counts[j] += 1
    return [
        (sums[j] / counts[j] if counts[j] > 0 else 0.0)
        for j in range(dims)
    ]


def covariance_matrix(vectors, epsilon=1e-6):
    """Sample covariance over complete valid rows + Tikhonov regularization.

    Only rows that are list/tuple of the common dimensionality with all
    entries valid are used. Unbiased (n-1) normalization; one valid row
    yields a zero matrix before regularization. epsilon on the diagonal
    guarantees invertibility for constant/collinear dims.
    """
    dims = None
    for r in vectors:
        if isinstance(r, (list, tuple)):
            dims = len(r)
            break
    if dims is None or dims == 0:
        raise ValueError("no valid data: cannot infer dimensionality")
    complete = []
    for r in vectors:
        if not isinstance(r, (list, tuple)):
            continue
        if len(r) != dims:
            continue
        if all(_is_valid_number(v) for v in r):
            complete.append([float(v) for v in r])
    if not complete:
        raise ValueError("no valid rows: all vectors contain None/NaN")
    n = len(complete)
    d = dims
    means = [sum(r[i] for r in complete) / n for i in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    if n > 1:
        for i in range(d):
            for j in range(i, d):
                s = sum(
                    (r[i] - means[i]) * (r[j] - means[j])
                    for r in complete
                ) / (n - 1)
                cov[i][j] = s
                cov[j][i] = s
    for i in range(d):
        cov[i][i] += epsilon
    return cov


def invert_matrix(m):
    """Matrix inverse via Gauss-Jordan elimination with partial pivoting.

    Raises ValueError("singular matrix...") if a pivot is ~0 even after
    regularization.
    """
    n = len(m)
    if n == 0 or any(len(row) != n for row in m):
        raise ValueError("singular matrix: expected non-empty square matrix")
    a = [list(map(float, row)) for row in m]
    inv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("singular matrix: pivot ~0, not invertible")
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            inv[col], inv[piv] = inv[piv], inv[col]
        pv = a[col][col]
        for j in range(n):
            a[col][j] /= pv
            inv[col][j] /= pv
        for r in range(n):
            if r != col:
                f = a[r][col]
                if f != 0.0:
                    for j in range(n):
                        a[r][j] -= f * a[col][j]
                        inv[r][j] -= f * inv[col][j]
    return inv


def mahalanobis(x, mean, inv):
    """Mahalanobis (d2, score): d2 >= 0, score = d2/(d2+k) in 0..1.

    d2 == 0 at the mean; score -> 1 as d2 -> infinity. Empty x yields
    (0.0, 0.0).
    """
    k = len(x)
    if k == 0:
        return (0.0, 0.0)
    diff = [float(x[i]) - float(mean[i]) for i in range(k)]
    d2 = 0.0
    for i in range(k):
        s = 0.0
        for j in range(k):
            s += inv[i][j] * diff[j]
        d2 += diff[i] * s
    if d2 < 0.0 and d2 > -1e-12:
        d2 = 0.0
    d2 = max(0.0, float(d2))
    score = d2 / (d2 + k) if math.isfinite(d2) else 1.0
    return (d2, score)


def detect_multivariate(vectors, threshold=0.8, epsilon=1e-6):
    """Flag rows with Mahalanobis score >= threshold (0..1, higher = worse).

    Rows with None/NaN/non-numeric/inf/wrong-length entries are skipped
    and counted, never raising. Returns {"anomalies": [{"index"
    (original row), "d2", "score", "causes" (top-2 dims by |diff|/std),
    "explanation"}], "skipped", "dims", "reason"}.
    """
    if not vectors:
        return {"anomalies": [], "skipped": 0, "dims": 0,
                "reason": "no data: empty input"}
    dims = None
    for r in vectors:
        if isinstance(r, (list, tuple)):
            dims = len(r)
            break
    if dims is None or dims == 0:
        skipped = len(list(vectors))
        return {"anomalies": [], "skipped": skipped, "dims": 0,
                "reason": "no data: no sequence rows found"}
    valid = []
    valid_idx = []
    skipped = 0
    for idx, r in enumerate(vectors):
        if (not isinstance(r, (list, tuple)) or len(r) != dims
                or not all(_is_valid_number(v) for v in r)):
            skipped += 1
            continue
        valid.append([float(v) for v in r])
        valid_idx.append(idx)
    if not valid:
        return {"anomalies": [], "skipped": skipped, "dims": dims,
                "reason": "no valid rows: all skipped (None/NaN or malformed)"}
    n = len(valid)
    means = [sum(r[i] for r in valid) / n for i in range(dims)]
    try:
        cov = covariance_matrix(valid, epsilon=epsilon)
        inv = invert_matrix(cov)
    except ValueError as e:
        return {"anomalies": [], "skipped": skipped, "dims": dims,
                "reason": "singular covariance: %s" % e}
    stds = [math.sqrt(cov[i][i]) if cov[i][i] > 0 else 1.0
            for i in range(dims)]
    anomalies = []
    for row, idx in zip(valid, valid_idx):
        d2, score = mahalanobis(row, means, inv)
        if score >= threshold:
            diffs = [abs(row[i] - means[i]) for i in range(dims)]
            z = [diffs[i] / stds[i] if stds[i] > 0 else 0.0
                 for i in range(dims)]
            ranked = sorted(range(dims), key=lambda i: (-z[i], i))
            causes = [i for i in ranked[:2] if z[i] > 1e-9]
            if causes:
                expl = ("row %d: multivariate score %.4f (d2=%.4f) "
                        "driven by dim(s) %s" % (idx, score, d2, causes))
            else:
                expl = ("row %d: multivariate score %.4f (d2=%.4f); "
                        "no single dominant dim" % (idx, score, d2))
            anomalies.append({"index": idx, "d2": d2, "score": score,
                              "causes": causes, "explanation": expl})
    reason = ("%d anomal%s among %d valid rows (skipped %d, dims %d, "
              "threshold %.3f)" % (
                  len(anomalies), "y" if len(anomalies) == 1 else "ies",
                  len(valid), skipped, dims, threshold))
    return {"anomalies": anomalies, "skipped": skipped, "dims": dims,
            "reason": reason}
