"""Pattern-level transitions: thin wrapper reusing markov logic."""
try:
    from nexora.prediction.markov import build_transition_matrix, predict_next
except ImportError:  # package-relative fallback
    try:
        from .markov import build_transition_matrix, predict_next
    except ImportError:
        build_transition_matrix = None
        predict_next = None


def record_pattern_sequence(pattern_ids):
    """Record a sequence of pattern ids; returns a markov matrix dict.

    P(next|current) semantics identical to markov.build_transition_matrix,
    applied to pattern-id tokens instead of raw labels.
    """
    seq = list(pattern_ids or [])
    if build_transition_matrix is not None:
        return build_transition_matrix(seq)
    states = sorted(set(seq), key=lambda x: str(x))
    counts, totals = {}, {}
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
        totals[a] = totals.get(a, 0) + 1
    probs = {k: (c / totals[k[0]]) for k, c in counts.items()}
    return {"states": states, "counts": counts, "probs": probs}


def predict_next_pattern(current_id, matrix, top_k=3):
    """Predict next pattern id after current_id. Same shape as markov.predict_next."""
    if predict_next is not None:
        return predict_next(current_id, matrix, top_k=top_k)
    if not matrix:
        return []
    counts = matrix.get("counts", {})
    cands, total = {}, 0
    for (a, b), c in counts.items():
        if a == current_id:
            cands[b] = cands.get(b, 0) + c
            total += c
    if total == 0:
        return []
    ranked = sorted(cands.items(), key=lambda kv: (-kv[1] / total, str(kv[0])))
    return [{"next": n, "probability": c / total,
             "evidence": ("Pattern '%s'->'%s' seen %d/%d times (%.1f%%)."
                          % (current_id, n, c, total, 100.0 * c / total))}
            for n, c in ranked[:max(1, int(top_k))]]
