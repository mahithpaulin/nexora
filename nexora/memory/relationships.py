"""Pattern co-occurrence relationships from label sequences.

Adjacency rule: for each pattern, find all start indices where its token
sequence occurs contiguously in label_seq. An occurrence covers
[start, start+len(seq)-1]. Occurrence of A ending exactly where
occurrence of B starts (end_A + 1 == start_B) counts one A->B transition.
Probabilities: followed_by[A][B] = count(A->B)/total outgoing from A;
preceded_by[B][A] = count(A->B)/total incoming to B (0..1). Self-transitions
allowed when genuinely adjacent repeats occur. Ordering is deterministic
(sorted by pid string).
"""


def _get(obj, key, default=None):
    """Read key from a dict or attribute from an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pid_seq(p):
    """Extract (pid, sequence) from a pattern dict/object."""
    pid = _get(p, "id", None)
    if pid is None:
        pid = _get(p, "pid", None)
    seq = _get(p, "sequence", None)
    if seq is None:
        seq = _get(p, "seq", None)
    if seq is None:
        seq = _get(p, "tokens", None)
    return pid, seq


def find_starts(label_seq, token_seq):
    """Start indices where token_seq occurs contiguously in label_seq.

    Comparison is string-based (str(a) == str(b)). Empty token_seq or
    non-sequence inputs return [].
    """
    if token_seq is None or label_seq is None:
        return []
    try:
        m = len(token_seq)
        n = len(label_seq)
    except TypeError:
        return []
    if m == 0 or n == 0 or m > n:
        return []
    try:
        toks = [str(t) for t in token_seq]
        labs = [str(t) for t in label_seq]
    except TypeError:
        return []
    starts = []
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if labs[i + j] != toks[j]:
                match = False
                break
        if match:
            starts.append(i)
    return starts


def build_relationships(label_seq, patterns):
    """Build adjacency transitions between patterns.

    Returns {pid: {"preceded_by": {other: prob}, "followed_by":
    {other: prob}, "evidence": {other: count}}}. Inner dicts are
    inserted in pid-string sorted order for determinism.
    """
    labels = list(label_seq) if label_seq is not None else []
    items = []
    for p in patterns or []:
        pid, seq = _pid_seq(p)
        if pid is None:
            continue
        seql = list(seq) if seq is not None else []
        items.append((pid, seql))
    pids_sorted = sorted([pid for pid, _ in items], key=lambda v: str(v))
    starts_at = {}
    ends_at = {}
    for pid, seq in items:
        if not seq:
            continue
        for s in find_starts(labels, seq):
            e = s + len(seq) - 1
            starts_at.setdefault(s, []).append(pid)
            ends_at.setdefault(e, []).append(pid)
    for v in starts_at.values():
        v.sort(key=str)
    for v in ends_at.values():
        v.sort(key=str)
    out_counts = {pid: {} for pid in pids_sorted}
    for e in sorted(ends_at):
        s2 = e + 1
        if s2 not in starts_at:
            continue
        for a in ends_at[e]:
            for b in starts_at[s2]:
                out_counts[a][b] = out_counts[a].get(b, 0) + 1
    out_total = {pid: sum(out_counts[pid].values()) for pid in pids_sorted}
    in_total = {pid: 0 for pid in pids_sorted}
    for a in pids_sorted:
        for b, c in out_counts[a].items():
            if b in in_total:
                in_total[b] += c
    rels = {}
    for pid in pids_sorted:
        followed = {}
        if out_total[pid] > 0:
            for other in sorted(out_counts[pid], key=str):
                followed[other] = out_counts[pid][other] / out_total[pid]
        preceded = {}
        incoming = {}
        for other in pids_sorted:
            c = out_counts.get(other, {}).get(pid, 0)
            if c > 0:
                incoming[other] = c
        if in_total[pid] > 0:
            for other in sorted(incoming, key=str):
                preceded[other] = incoming[other] / in_total[pid]
        evidence = {}
        others = set(out_counts[pid]) | set(incoming)
        for other in sorted(others, key=str):
            if other == pid:
                evidence[other] = out_counts[pid].get(pid, 0)
            else:
                evidence[other] = (out_counts[pid].get(other, 0)
                                   + out_counts.get(other, {}).get(pid, 0))
        rels[pid] = {"preceded_by": preceded, "followed_by": followed,
                     "evidence": evidence}
    return rels


def attach_relationships(patterns, rels):
    """Attach top-3 relationships to each pattern; return the list.

    Sets relationships = {"commonly_preceded_by": [...top-3 pids...],
    "commonly_followed_by": [...], "probs": {"preceded_by": {...},
    "followed_by": {...}}}. Top-3 by probability desc, ties by pid
    string. Mutates dict patterns in place (or sets the attribute on
    objects). Missing entries yield empty lists/dicts.
    """
    rels = rels or {}
    for p in patterns or []:
        pid, _ = _pid_seq(p)
        entry = rels.get(pid, None)
        if entry is None:
            for k in rels:
                if str(k) == str(pid):
                    entry = rels[k]
                    break
        if not isinstance(entry, dict):
            entry = {}
        prec = entry.get("preceded_by", {}) or {}
        foll = entry.get("followed_by", {}) or {}
        top_prec = sorted(prec, key=lambda o: (-prec[o], str(o)))[:3]
        top_foll = sorted(foll, key=lambda o: (-foll[o], str(o)))[:3]
        relationships = {
            "commonly_preceded_by": list(top_prec),
            "commonly_followed_by": list(top_foll),
            "probs": {"preceded_by": dict(prec),
                      "followed_by": dict(foll)},
        }
        if isinstance(p, dict):
            p["relationships"] = relationships
        else:
            try:
                setattr(p, "relationships", relationships)
            except (AttributeError, TypeError):
                pass
    return patterns
