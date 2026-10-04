"""Pattern evolution tracking: snapshots, drift scoring, trend tracking.

Drift score 0..1 (higher = more changed): 0 = identical comparable
values, 1 = fully changed (relative change >= 100% on average).
"""

import math


def _get(obj, key, default=None):
    """Read key from a dict or attribute from an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_num(v):
    """True for finite int/float excluding bool/None/NaN/inf."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    return False


def snapshot(pattern):
    """Capture a comparable snapshot: id, type, features, confidence, frequency, state."""
    feats = _get(pattern, "features", {})
    if not isinstance(feats, dict):
        feats = {}
    else:
        feats = dict(feats)
    return {
        "id": _get(pattern, "id", None),
        "type": _get(pattern, "type", None),
        "features": feats,
        "confidence": _get(pattern, "confidence", None),
        "frequency": _get(pattern, "frequency", None),
        "state": _get(pattern, "state", None),
    }


def _drift(old_f, new_f, tolerance=1e-9):
    """Per-key drift = min(1, abs(new-old)/(abs(old)+tolerance))."""
    return min(1.0, abs(new_f - old_f) / (abs(old_f) + tolerance))


def drift_score(old, new, tolerance=1e-9):
    """Drift between two snapshots in 0..1 (higher = more changed).

    Compares numeric top-level fields (confidence, frequency) and
    numeric feature values present in both. Added/removed keys are
    ignored in the mean but listed in reason. 0.0 when no comparable
    keys exist.
    """
    old_feats = _get(old, "features", {}) or {}
    new_feats = _get(new, "features", {}) or {}
    if not isinstance(old_feats, dict):
        old_feats = {}
    if not isinstance(new_feats, dict):
        new_feats = {}
    pairs = []
    oc = _get(old, "confidence", None)
    nc = _get(new, "confidence", None)
    if _is_num(oc) and _is_num(nc):
        pairs.append(("confidence", float(oc), float(nc)))
    of = _get(old, "frequency", None)
    nf = _get(new, "frequency", None)
    if _is_num(of) and _is_num(nf):
        pairs.append(("frequency", float(of), float(nf)))
    for k in sorted(set(old_feats) & set(new_feats), key=str):
        ov, nv = old_feats[k], new_feats[k]
        if _is_num(ov) and _is_num(nv):
            pairs.append((k, float(ov), float(nv)))
    shifts = {}
    drifts = []
    for k, ov, nv in pairs:
        d = _drift(ov, nv, tolerance)
        drifts.append(d)
        shifts[k] = {"old": ov, "new": nv, "rel_change": d}
    score = sum(drifts) / len(drifts) if drifts else 0.0
    added = sorted(set(new_feats) - set(old_feats), key=str)
    removed = sorted(set(old_feats) - set(new_feats), key=str)
    reason = ("compared %d keys, mean drift %.4f; added %s; removed %s"
              % (len(pairs), score, added, removed))
    return {"score": float(score), "shifts": shifts, "reason": reason}


def track(snapshots):
    """Trend over consecutive snapshots.

    Needs >= 2 snapshots else "unknown". All scores < 0.1 -> "stable";
    monotonically non-decreasing with last >= 0.3 -> "drifting";
    monotonically non-increasing -> "converging"; else "oscillating"
    (checked in that order, so equal scores >= 0.3 report "drifting").
    Returns {"trend", "scores", "reason"}.
    """
    if snapshots is None or len(snapshots) < 2:
        return {"trend": "unknown", "scores": [],
                "reason": "need >= 2 snapshots, got %s"
                % (0 if snapshots is None else len(snapshots))}
    scores = []
    for i in range(len(snapshots) - 1):
        scores.append(drift_score(snapshots[i], snapshots[i + 1])["score"])
    if all(s < 0.1 for s in scores):
        trend = "stable"
    elif (all(scores[i + 1] >= scores[i] for i in range(len(scores) - 1))
            and scores[-1] >= 0.3):
        trend = "drifting"
    elif all(scores[i + 1] <= scores[i] for i in range(len(scores) - 1)):
        trend = "converging"
    else:
        trend = "oscillating"
    reason = "trend %s from %d consecutive drift scores %s" % (
        trend, len(scores),
        "[" + ", ".join("%.4f" % s for s in scores) + "]")
    return {"trend": trend, "scores": scores, "reason": reason}
