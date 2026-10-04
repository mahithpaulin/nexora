"""Windowed-embedding clustering (stdlib only, deterministic, random.Random(0)).
Lift 1-D series via sliding windows (time-delay embedding), then kmeans /
dbscan / agglomerative. Rows with None/NaN/inf/non-numeric are skipped, never
crash. Agglomerative holds an O(n^2) point-distance matrix (v0.1: small n only)."""
import math
import random


def _ok(v):
    return v is not None and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _clean(vectors):
    """Drop invalid rows; raise ValueError if none remain or ragged."""
    if vectors is None:
        raise ValueError("vectors is empty")
    clean = []
    for r in list(vectors):
        if r is None:
            continue
        try:
            items = list(r)
        except TypeError:
            continue
        if not items:
            continue
        fv = []
        good = True
        for v in items:
            if not _ok(v):
                good = False
                break
            fv.append(float(v))
        if good:
            clean.append(fv)
    if not clean:
        raise ValueError("vectors is empty (or all rows invalid)")
    d = len(clean[0])
    if any(len(r) != d for r in clean):
        raise ValueError("ragged input: all vectors need equal length")
    return clean


def _sq(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _dist(a, b):
    return math.sqrt(_sq(a, b))


def _mean(rows):
    d = len(rows[0])
    m = len(rows)
    return [sum(r[j] for r in rows) / m for j in range(d)]


def _auto_eps(vecs):
    if len(vecs) < 2:
        return 1.0
    tot = sum(min(_dist(p, q) for j, q in enumerate(vecs) if j != i) for i, p in enumerate(vecs))
    avg = tot / len(vecs)
    return float(avg * 1.5 + 1e-9) if math.isfinite(avg) and avg > 0.0 else 1.0


def windows(values, size, stride=1):
    """Inputs: values (1-D series of numbers/None/NaN), size (int>=1), stride (int>=1). Outputs: (vectors, start_indices); vectors are float lists of length size; any window with None/NaN/inf/non-numeric is skipped. Scores: none (no 0..1 score). Raises ValueError if size<1 or stride<1."""
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be an int >= 1")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be an int >= 1")
    if values is None:
        return ([], [])
    vals = list(values)
    vecs, starts = [], []
    for s in range(0, len(vals) - size + 1, stride):
        fv = []
        good = True
        for v in vals[s:s + size]:
            if not _ok(v):
                good = False
                break
            fv.append(float(v))
        if good:
            vecs.append(fv)
            starts.append(s)
    return (vecs, starts)


def kmeans(vectors, k, max_iter=100):
    """Inputs: vectors (equal-length numeric vecs; invalid rows skipped), k (int 1..n), max_iter (int>=1). Outputs: {labels, centroids, inertia (float>=0, lower=tighter, unbounded), iterations (int), k, reason (converged|max_iter)}. k-means++ init with random.Random(0), Lloyd iterations, empty cluster -> farthest point (ties: smallest index). Deterministic. Scores: no 0..1 score. Raises ValueError on empty/all-invalid/ragged input, k<1, k>n, max_iter<1."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an int >= 1")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be an int >= 1")
    c = _clean(vectors)
    n = len(c)
    if k > n:
        raise ValueError("k must be <= number of (valid) vectors")
    d = len(c[0])
    rng = random.Random(0)
    cents = [list(c[rng.randrange(n)])]
    while len(cents) < k:
        d2 = [min(_sq(p, q) for q in cents) for p in c]
        tot = sum(d2)
        if tot <= 0.0 or not math.isfinite(tot):
            cents.append(list(c[rng.randrange(n)]))
            continue
        r = rng.random() * tot
        acc, idx = 0.0, n - 1
        for i, v in enumerate(d2):
            acc += v
            if r <= acc:
                idx = i
                break
        cents.append(list(c[idx]))
    labels, prev, iters, reason = [-1] * n, None, 0, ""
    for it in range(1, max_iter + 1):
        iters = it
        labels = [min(range(k), key=lambda ci: _sq(p, cents[ci])) for p in c]
        if prev is not None and labels == prev:
            reason = f"converged after {iters} iteration(s)"
            break
        prev = list(labels)
        sums = [[0.0] * d for _ in range(k)]
        cnts = [0] * k
        for li, p in zip(labels, c):
            cnts[li] += 1
            for j in range(d):
                sums[li][j] += p[j]
        new = [[s / cnts[i] for s in sums[i]] if cnts[i] else None for i in range(k)]
        for i in range(k):
            if new[i] is None:
                ref = [q for q in new if q is not None] or cents
                # farthest point, smallest index on ties
                bd, bi = -1.0, 0
                for t, p in enumerate(c):
                    md = min(_sq(p, q) for q in ref)
                    if md > bd:
                        bd, bi = md, t
                new[i] = list(c[bi])
        cents = list(new)
        if it == max_iter:
            reason = f"max_iter ({max_iter}) reached without full convergence"
    inertia = sum(_sq(p, cents[li]) for p, li in zip(c, labels))
    return {"labels": labels, "centroids": [list(q) for q in cents], "inertia": float(inertia), "iterations": iters, "k": k, "reason": reason}


def dbscan(vectors, eps, min_pts=3):
    """Inputs: vectors (invalid rows skipped), eps (finite float>0 radius), min_pts (int>=1 core threshold). Outputs: {labels (-1=noise), n_clusters (int>=0), n_noise (int>=0), reason (str)}. Index-ordered expansion, deterministic. Scores: no 0..1 score. Raises ValueError on empty/all-invalid/ragged input, bad eps, or min_pts<1."""
    if not isinstance(min_pts, int) or min_pts < 1:
        raise ValueError("min_pts must be an int >= 1")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be a finite float > 0")
    eps = float(eps)
    c = _clean(vectors)
    n = len(c)
    e2 = eps * eps
    nb = [[j for j in range(n) if _sq(c[i], c[j]) <= e2] for i in range(n)]
    core = [len(v) >= min_pts for v in nb]
    labels, seen, cid = [-1] * n, [False] * n, 0
    for i in range(n):
        if seen[i]:
            continue
        seen[i] = True
        if not core[i]:
            continue
        labels[i] = cid
        seeds = [j for j in nb[i] if j != i]
        queued, q = set(seeds), 0
        while q < len(seeds):
            j = seeds[q]
            q += 1
            if not seen[j]:
                seen[j] = True
                if core[j]:
                    for t in nb[j]:
                        if t not in queued:
                            queued.add(t)
                            seeds.append(t)
            if labels[j] == -1:
                labels[j] = cid
        cid += 1
    nz = sum(1 for v in labels if v == -1)
    return {"labels": labels, "n_clusters": cid, "n_noise": nz, "reason": f"done: {cid} cluster(s), {nz} noise point(s)"}


def agglomerative(vectors, n_clusters):
    """Inputs: vectors (invalid rows skipped), n_clusters (int 1..n). Single-linkage bottom-up; holds full n x n distance matrix -> O(n^2) memory/time (v0.1 limitation, small n only). Outputs: {labels (0..n_clusters-1 ordered by smallest member), n_clusters, merges ([(a,b,dist)] all n-n_clusters merges in order; a,b are cluster ids), reason (str)}. Ties by (dist,min_id,max_id). Deterministic. Scores: no 0..1 score. Raises ValueError on empty/all-invalid/ragged input or bad n_clusters."""
    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("n_clusters must be an int >= 1")
    c = _clean(vectors)
    n = len(c)
    if n_clusters > n:
        raise ValueError("n_clusters must be <= number of (valid) vectors")
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = _dist(c[i], c[j])
            dm[i][j] = v
            dm[j][i] = v
    cls = {i: {i} for i in range(n)}
    act, nxt, merges = set(range(n)), n, []
    while len(act) > n_clusters:
        act_s = sorted(act)
        best, pair = None, None
        for x in range(len(act_s)):
            for y in range(x + 1, len(act_s)):
                a, b = act_s[x], act_s[y]
                dd = min(dm[i][j] for i in cls[a] for j in cls[b])
                key = (dd, min(a, b), max(a, b))
                if best is None or key < best:
                    best, pair = key, (a, b)
        a, b = pair
        cls[nxt] = cls[a] | cls[b]
        act.remove(a)
        act.remove(b)
        act.add(nxt)
        merges.append((a, b, float(best[0])))
        nxt += 1
    order = sorted(act, key=lambda v: min(cls[v]))
    mp = {}
    for li, v in enumerate(order):
        for m in cls[v]:
            mp[m] = li
    return {"labels": [mp[i] for i in range(n)], "n_clusters": n_clusters, "merges": merges, "reason": f"done: merged to {n_clusters} cluster(s) ({len(merges)} merge(s))"}


def find_regimes(values, size=8, k=2, method="kmeans"):
    """Inputs: values (1-D series; windows with None/NaN/inf skipped), size (int>=1 window len), k (int>=1 target clusters; ignored for dbscan; clamped to #windows for kmeans/agglomerative), method (kmeans|dbscan|agglomerative; dbscan auto-eps=1.5x mean nearest-neighbour dist, min_pts=3). Outputs: list sorted by cluster id of pattern-dicts with keys exactly {id None, type regime, features {method,cluster,size,count,support,centroid}, sequence (centroid), frequency (count), occurrences ([window starts]), confidence (support)}. [] if no valid windows (or all windows are dbscan noise). Deterministic. Scores: support/confidence in 0..1 (count/#windows, noise windows included in the denominator but never emitted). Raises ValueError on size<1, k<1, unknown method, and (v2) when method != "kmeans" and #windows > 1000 (suggests kmeans).

    v2 notes:
    - method="dbscan": noise label -1 is filtered out — noise windows are
      counted in dbscan's reason string (via n_noise) but never emitted as
      regime patterns. kmeans/agglomerative paths are unchanged.
    - Size guard: method != "kmeans" with > 1000 windows raises ValueError
      suggesting kmeans, because dbscan builds an O(n^2) pairwise
      neighborhood and agglomerative holds an O(n^2) distance matrix with
      O(n^3) single-linkage merging — both infeasible at large n, while
      kmeans stays O(n*k*iter). Determinism preserved throughout.
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be an int >= 1")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an int >= 1")
    if method not in ("kmeans", "dbscan", "agglomerative"):
        raise ValueError("method must be kmeans, dbscan or agglomerative")
    vecs, starts = windows(values, size)
    if not vecs:
        return []
    if method != "kmeans" and len(vecs) > 1000:
        raise ValueError(
            f"too many windows ({len(vecs)}) for method={method!r} "
            "with O(n^2)/O(n^3) cost (dbscan pairwise neighborhoods, "
            "agglomerative distance matrix + merging); use method='kmeans'"
        )
    if method == "kmeans":
        ke = min(k, len(vecs))
        r = kmeans(vecs, ke)
        labs, cents = r["labels"], {i: r["centroids"][i] for i in range(ke)}
    elif method == "agglomerative":
        ke = min(k, len(vecs))
        r = agglomerative(vecs, ke)
        labs = r["labels"]
        cents = {t: _mean([v for v, lb in zip(vecs, labs) if lb == t]) for t in range(ke)}
    else:
        r = dbscan(vecs, _auto_eps(vecs), 3)
        labs = r["labels"]
        cents = {t: _mean([v for v, lb in zip(vecs, labs) if lb == t]) for t in sorted(set(labs)) if t != -1}
    tot = len(vecs)
    out = []
    for t in sorted(set(labs)):
        if method == "dbscan" and t == -1:
            continue  # noise reported in dbscan reason, not as patterns
        idx = [i for i, lb in enumerate(labs) if lb == t]
        cnt = len(idx)
        sup = cnt / tot
        cent = list(cents[t])
        out.append({"id": None, "type": "regime", "features": {"method": method, "cluster": int(t), "size": size, "count": cnt, "support": float(sup), "centroid": list(cent)}, "sequence": list(cent), "frequency": cnt, "occurrences": [starts[i] for i in idx], "confidence": float(sup)})
    return out
