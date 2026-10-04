"""First-order Markov chain over discrete labels. Stdlib only, deterministic."""


def build_transition_matrix(label_seq):
    """Build bigram counts and conditional probs. Returns {states, counts, probs}.

    probs[(a, b)] = P(b|a) = count(a->b) / total_from(a), in 0..1.
    states is the sorted unique label list. Empty input -> empty matrix.
    """
    seq = [s for s in (label_seq or [])]
    states = sorted(set(seq), key=lambda x: str(x))
    counts = {}
    totals = {}
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
        totals[a] = totals.get(a, 0) + 1
    probs = {}
    for (a, b), c in counts.items():
        probs[(a, b)] = c / totals[a] if totals[a] else 0.0
    return {"states": states, "counts": counts, "probs": probs}


def predict_next(current, matrix, top_k=3):
    """P(next|current) = count(current->next)/total(current). Returns ranked list.

    Each item: {"next", "probability" (0..1), "evidence" (counts cited)}.
    Empty when current has no recorded outgoing transitions.
    """
    if not matrix:
        return []
    counts = matrix.get("counts", {})
    cands = {}
    total = 0
    for (a, b), c in counts.items():
        if a == current:
            cands[b] = cands.get(b, 0) + c
            total += c
    if total == 0:
        return []
    ranked = sorted(cands.items(), key=lambda kv: (-kv[1] / total, str(kv[0])))
    out = []
    for nxt, c in ranked[:max(1, int(top_k))]:
        p = c / total
        out.append({
            "next": nxt,
            "probability": p,
            "evidence": ("Observed %d transition(s) '%s'->'%s' out of %d from '%s' (%.1f%%)."
                         % (c, current, nxt, total, current, 100.0 * p)),
        })
    return out
