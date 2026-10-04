"""PCA via power iteration + deflation (stdlib only).

``pca()`` mean-centres, builds the population covariance (divide by n),
and extracts eigenpairs with power iteration (fixed shifted 1..d init per
component + Gram-Schmidt vs found components, 200 max iterations, 1e-10
tolerance) + deflation; tiny negative eigenvalues are
clamped to 0.0. ``project()`` maps vectors onto stored components;
``reconstruction_error()`` reports mean squared error of the
project-then-reconstruct round trip. Rows with None/NaN/inf or
non-numeric entries are skipped (never crash). Deterministic.
"""

from __future__ import annotations

import math


def _is_finite_number(v) -> bool:
    return v is not None and isinstance(v, (int, float)) \
        and not isinstance(v, bool) and math.isfinite(float(v))


def _clean_matrix(vectors) -> list[list[float]]:
    """Drop invalid rows; raise ValueError if none remain or ragged."""
    if vectors is None:
        raise ValueError("vectors is empty")
    rows = list(vectors)
    clean: list[list[float]] = []
    for r in rows:
        if r is None:
            continue
        try:
            items = list(r)
        except TypeError:
            continue
        if not items:
            continue
        fv: list[float] = []
        ok = True
        for v in items:
            if not _is_finite_number(v):
                ok = False
                break
            fv.append(float(v))
        if ok:
            clean.append(fv)
    if not clean:
        raise ValueError("vectors is empty (or all rows invalid)")
    dim = len(clean[0])
    if any(len(r) != dim for r in clean):
        raise ValueError("ragged input: all vectors need equal length")
    return clean


def _mat_vec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]


def pca(vectors, n_components=2) -> dict:
    """Principal component analysis (power iteration + deflation).

    Mean-centres, builds the population covariance (divide by n), then
    power iteration (fixed shifted init + Gram-Schmidt, 200 max iters,
    1e-10 tolerance) with deflation. Tiny negative eigenvalues clamped
    to 0.0. Returns {"components", "explained_variance",
    "explained_ratio" (0..1 each, sums to ~1), "mean", "n_components",
    "reason"}. Deterministic.
    """
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be an int >= 1")
    clean = _clean_matrix(vectors)
    n = len(clean)
    dim = len(clean[0])
    nc = min(n_components, dim)
    mean = [sum(r[j] for r in clean) / n for j in range(dim)]
    centred = [[r[j] - mean[j] for j in range(dim)] for r in clean]
    cov = [[0.0] * dim for _ in range(dim)]
    for row in centred:
        for i in range(dim):
            for j in range(i, dim):
                cov[i][j] += row[i] * row[j]
    for i in range(dim):
        for j in range(i, dim):
            cov[i][j] /= n
            cov[j][i] = cov[i][j]
    work = [row[:] for row in cov]
    components: list[list[float]] = []
    variances: list[float] = []
    for ci in range(nc):
        b = [float(i + 1 + ci) for i in range(dim)]
        for pv in components:  # Gram-Schmidt: keep inits out of found subspace
            d = sum(b[i] * pv[i] for i in range(dim))
            b = [b[i] - d * pv[i] for i in range(dim)]
        nb = math.sqrt(sum(x * x for x in b))
        if nb < 1e-12:  # init lay in found subspace: first orthogonal axis
            for a in range(dim):
                e = [0.0] * dim
                e[a] = 1.0
                for pv in components:
                    d = sum(e[i] * pv[i] for i in range(dim))
                    e = [e[i] - d * pv[i] for i in range(dim)]
                ne = math.sqrt(sum(x * x for x in e))
                if ne > 1e-9:
                    b = [x / ne for x in e]
                    nb = 1.0
                    break
        b = [x / nb for x in b] if nb >= 1e-12 else b
        for _ in range(200):
            ab = _mat_vec(work, b)
            norm = math.sqrt(sum(x * x for x in ab))
            if norm < 1e-12:
                break
            ab = [x / norm for x in ab]
            dp = math.sqrt(sum((ab[i] - b[i]) ** 2 for i in range(dim)))
            dn = math.sqrt(sum((ab[i] + b[i]) ** 2 for i in range(dim)))
            b = ab
            if min(dp, dn) < 1e-10:
                break
        ab = _mat_vec(work, b)
        lam = sum(b[i] * ab[i] for i in range(dim))
        if lam < 0.0:
            lam = 0.0
        variances.append(float(lam))
        components.append(list(b))
        for i in range(dim):
            for j in range(dim):
                work[i][j] -= lam * b[i] * b[j]
    total = sum(variances)
    if total > 0.0:
        ratio = [v / total for v in variances]
    else:
        ratio = [0.0] * len(variances)
    return {"components": components, "explained_variance": variances,
            "explained_ratio": ratio, "mean": mean,
            "n_components": nc,
            "reason": f"done: {nc} component(s) from {n} vector(s) "
                      f"(dim {dim})"}


def project(vectors, model) -> list[list[float]]:
    """Project vectors onto stored PCA components (invalid rows skipped)."""
    try:
        mean = [float(x) for x in model["mean"]]
        comps = [[float(x) for x in c] for c in model["components"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("model must be a pca() dict with mean/components")
    dim = len(mean)
    out: list[list[float]] = []
    for r in (vectors or []):
        if r is None:
            continue
        try:
            fv = [float(v) for v in list(r)]
        except (TypeError, ValueError):
            continue
        if len(fv) != dim or any(not math.isfinite(x) for x in fv):
            continue
        centred = [fv[j] - mean[j] for j in range(dim)]
        out.append([sum(centred[j] * c[j] for j in range(dim))
                    for c in comps])
    return out


def reconstruction_error(vectors, model) -> float:
    """Mean squared error of the project-then-reconstruct round trip (>= 0)."""
    try:
        mean = [float(x) for x in model["mean"]]
        comps = [[float(x) for x in c] for c in model["components"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("model must be a pca() dict with mean/components")
    dim = len(mean)
    valid: list[list[float]] = []
    for r in (vectors or []):
        if r is None:
            continue
        try:
            fv = [float(v) for v in list(r)]
        except (TypeError, ValueError):
            continue
        if len(fv) != dim or any(not math.isfinite(x) for x in fv):
            continue
        valid.append(fv)
    if not valid or not comps:
        return 0.0
    se = 0.0
    for fv in valid:
        centred = [fv[j] - mean[j] for j in range(dim)]
        coefs = [sum(centred[j] * c[j] for j in range(dim)) for c in comps]
        for j in range(dim):
            recon = mean[j] + sum(coefs[t] * comps[t][j]
                                  for t in range(len(comps)))
            se += (fv[j] - recon) ** 2
    return float(se / (len(valid) * dim))
