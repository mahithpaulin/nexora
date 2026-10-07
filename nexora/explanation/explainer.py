"""Template-based explanations that always cite numbers."""


def _get(p, key, default=None):
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def explain_discovery(pattern):
    """One-sentence discovery rationale citing id, type, frequency, confidence."""
    pid = _get(pattern, "id", "?")
    typ = _get(pattern, "type", "unknown")
    freq = _get(pattern, "frequency", 0)
    occ = _get(pattern, "occurrences", []) or []
    conf = _get(pattern, "confidence", 0.0) or 0.0
    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        conf_f = 0.0
    feats = _get(pattern, "features", {})
    return ("Pattern %s was identified because type '%s' recurred %s time(s) across %d "
            "occurrence(s) (features=%s) with confidence %.2f."
            % (pid, typ, freq, len(occ), feats, conf_f))


def explain_match(obs, pattern_id, similarity, matched_features, matched=None, threshold=0.5):
    """Match rationale citing similarity % and matched feature names.

    The text is generated from the actual verdict: matched=True says the
    observation matched, matched=False says it did not (citing the
    threshold), matched=None keeps the legacy neutral wording for
    callers that do not pass a verdict.
    """
    try:
        sim = float(similarity)
    except (TypeError, ValueError):
        sim = 0.0
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        thr = 0.5
    feats = list(matched_features or [])
    val = obs.get("value", obs.get("label")) if isinstance(obs, dict) else obs
    if matched is True:
        return ("Observation %s matched pattern %s with similarity %.1f%%; "
                "matched features: %s." % (val, pattern_id, 100.0 * sim, feats or "none"))
    if matched is False:
        return ("Observation %s did not match pattern %s: similarity %.1f%% is below "
                "the %.1f%% match threshold; overlapping features: %s."
                % (val, pattern_id, 100.0 * sim, 100.0 * thr, feats or "none"))
    return ("Observation %s compared against pattern %s with similarity %.1f%%; "
            "overlapping features: %s." % (val, pattern_id, 100.0 * sim, feats or "none"))


def explain_anomaly(anomaly):
    """Anomaly rationale citing index, kind, z-score, score, causes."""
    kind = _get(anomaly, "kind", "unknown")
    idx = _get(anomaly, "index", "?")
    val = _get(anomaly, "value", "?")
    z = _get(anomaly, "z", 0.0) or 0.0
    try:
        zf = float(z)
    except (TypeError, ValueError):
        zf = 0.0
    score = _get(anomaly, "score", 0.0) or 0.0
    try:
        sf = float(score)
    except (TypeError, ValueError):
        sf = 0.0
    causes = _get(anomaly, "causes", []) or []
    base = "Anomaly at index %s (value %s) of kind '%s' scored %.2f" % (idx, val, kind, sf)
    if kind == "statistical":
        base += " with z=%.2f" % zf
    if causes:
        base += " because " + "; ".join(str(c) for c in causes)
    return base + "."


def explain_prediction(pred):
    """Prediction rationale citing next state(s) and probabilities."""
    if isinstance(pred, list):
        if not pred:
            return "No prediction: current state has no recorded outgoing transitions."
        parts = []
        for p in pred:
            nxt = p.get("next", "?") if isinstance(p, dict) else p
            prob = 0.0
            if isinstance(p, dict):
                try:
                    prob = float(p.get("probability", 0.0))
                except (TypeError, ValueError):
                    prob = 0.0
            parts.append("'%s' (%.1f%%)" % (nxt, 100.0 * prob))
        return "Predicted next: " + ", ".join(parts) + "."
    if isinstance(pred, dict):
        nxt = pred.get("next", "?")
        try:
            prob = float(pred.get("probability", 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        ev = pred.get("evidence", "")
        return "Predicted next '%s' with probability %.3f. %s" % (nxt, prob, ev)
    return "Prediction: %s." % (pred,)


def summarize_result(result):
    """Count-based summary of a result dict plus its own explanation text."""
    if not isinstance(result, dict):
        return "Result: %s." % (result,)
    bits = []
    if "patterns" in result:
        try:
            bits.append("%d pattern(s)" % len(result.get("patterns") or []))
        except TypeError:
            pass
    if "matches" in result:
        bits.append("%d match(es)" % len(result.get("matches") or []))
    if "anomalies" in result:
        bits.append("%d anomalie(s)" % len(result.get("anomalies") or []))
    if "predictions" in result:
        bits.append("%d prediction(s)" % len(result.get("predictions") or []))
    if "count" in result and "patterns" not in result:
        bits.append("count=%s" % result.get("count"))
    head = ("Result summary: " + ", ".join(bits) + ".") if bits else "Result summary: empty."
    extra = result.get("explanation") or result.get("reason") or ""
    return (head + " " + str(extra)).strip()
