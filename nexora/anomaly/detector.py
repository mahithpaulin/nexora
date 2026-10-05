"""Statistical + sequence anomaly detection. Stdlib only, deterministic."""
import math


def _get(p, key, default=None):
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _rows(values_or_rows):
    out = []
    seq = list(values_or_rows) if values_or_rows is not None else []
    for i, item in enumerate(seq):
        if isinstance(item, dict):
            idx = item.get("index", i)
            val = item.get("value", item.get("label"))
            lab = item.get("label", None)
            if lab is None:
                lab = str(val)
            out.append({"index": idx, "value": val, "label": str(lab)})
        else:
            out.append({"index": i, "value": item, "label": str(item)})
    return out


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _known_sequences(patterns):
    bigrams = set()
    expected = {}
    if not patterns:
        return bigrams, expected
    for p in patterns:
        seq = _get(p, "sequence", []) or []
        labs = [str(x) for x in seq]
        feats = _get(p, "features", {}) or {}
        if not labs and isinstance(feats, dict) and "sequence" in feats:
            try:
                labs = [str(x) for x in (feats.get("sequence") or [])]
            except TypeError:
                labs = []
        for a, b in zip(labs, labs[1:]):
            bigrams.add((a, b))
            expected.setdefault(a, set()).add(b)
    return bigrams, expected


def detect(values_or_rows, stats=None, z_threshold=3.0, patterns=None):
    """Detect anomalies. Returns list of {index,value,z,score,kind,causes,explanation}.

    Kinds: "statistical" (|z| >= z_threshold, score = min(1, |z|/(2*threshold))),
    "missing_transition" (seen predecessor, unseen pairing, score 0.55),
    "novel_sequence" (unseen bigram with unseen predecessor, score 0.60).
    Score is 0..1 (higher = more anomalous). Every hit carries a
    human-readable explanation; context comes from the passed stats dict
    (mean/stdev) and known-pattern bigrams.

    v2: z_threshold <= 0 raises ValueError (previously silently reset to
    3.0). Non-numeric thresholds still fall back to 3.0.
    """
    stats = stats or {}
    if not isinstance(stats, dict):
        stats = {"mean": getattr(stats, "mean", 0.0), "stdev": getattr(stats, "stdev", 0.0)}
    mean = stats.get("mean", 0.0)
    stdev = stats.get("stdev", 0.0)
    try:
        mean = float(mean)
    except (TypeError, ValueError):
        mean = 0.0
    try:
        stdev = float(stdev)
    except (TypeError, ValueError):
        stdev = 0.0
    try:
        zt = float(z_threshold)
    except (TypeError, ValueError):
        zt = 3.0
    if zt <= 0:
        raise ValueError("z_threshold must be positive, got %r" % (z_threshold,))

    rows = _rows(values_or_rows)
    known_bi, expected = _known_sequences(patterns)
    labels = [r["label"] for r in rows]
    out = []

    for r in rows:
        idx, val = r["index"], r["value"]
        if _is_num(val) and stdev > 0:
            z = (float(val) - mean) / stdev
            if abs(z) >= zt:
                score = min(1.0, abs(z) / (zt * 2.0))
                out.append({
                    "index": idx, "value": val, "z": z, "score": score,
                    "kind": "statistical",
                    "causes": ["z-score %.2f exceeds threshold %.2f" % (z, zt)],
                    "explanation": ("Statistical outlier at index %s: value %s deviates "
                                    "z=%.2f from mean %.3f (stdev %.3f, threshold %.2f), score %.2f."
                                    % (idx, val, z, mean, stdev, zt, score)),
                })

    # Sequence checks over label bigrams. Skipped when the series is
    # constant (a flat series has no novelty by definition — D2) or when
    # no known bigrams exist. z is None here: no z-score was measured.
    if labels and known_bi and len(set(labels)) > 1:
        for i in range(len(labels) - 1):
            a, b = labels[i], labels[i + 1]
            if (a, b) in known_bi:
                continue
            idx = rows[i + 1]["index"]
            val = rows[i + 1]["value"]
            if a in expected:
                exp = sorted(expected[a])
                out.append({
                    "index": idx, "value": val, "z": None, "score": 0.55,
                    "kind": "missing_transition",
                    "causes": ["transition %s->%s never seen; expected one of %s" % (a, b, exp)],
                    "explanation": ("Missing expected transition at index %s: observed %s->%s, "
                                    "but '%s' was previously seen going to %s; score 0.55." % (idx, a, b, a, exp)),
                })
            else:
                out.append({
                    "index": idx, "value": val, "z": None, "score": 0.60,
                    "kind": "novel_sequence",
                    "causes": ["bigram (%s, %s) unseen in %d known pattern(s)" % (a, b, len(patterns or []))],
                    "explanation": ("Novel sequence at index %s: bigram '%s'->'%s' never appeared in "
                                    "known patterns; score 0.60." % (idx, a, b)),
                })
    out.sort(key=lambda d: (d["index"], d["kind"]))
    return out
