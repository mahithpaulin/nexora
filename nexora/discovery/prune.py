"""Closed/maximal sequential-pattern pruning (stdlib only).

Mining enumerates every frequent n-gram, so ``A, B, AB, ABC`` all get
reported even when they carry identical information. A pattern is
*closed* when no proper super-sequence has the same support-count;
*maximal* when no proper super-sequence is frequent at all. Dropping
non-closed patterns removes pure redundancy: the shorter pattern's
occurrences add nothing the longer one does not already say.

All functions are deterministic and never mutate their inputs.
"""

from __future__ import annotations


def norm_key(seq) -> tuple:
    """Str-normalized tuple key for a pattern sequence."""
    try:
        return tuple(str(x) for x in (seq or []))
    except TypeError:
        return (str(seq),)


def is_contiguous_subsequence(short, long) -> bool:
    """True when tuple ``short`` occurs contiguously inside ``long``."""
    s, lo = tuple(short), tuple(long)
    if not s or len(s) > len(lo):
        return False
    if len(s) == len(lo):
        return s == lo
    return any(lo[i:i + len(s)] == s for i in range(len(lo) - len(s) + 1))


def count_of(pattern) -> int:
    """Support-count of a pattern dict (``frequency`` wins, then features)."""
    if not isinstance(pattern, dict):
        return 0
    for key in ("frequency", "support", "count"):
        if key not in pattern or pattern[key] is None:
            continue
        try:
            return int(pattern[key])
        except (TypeError, ValueError):
            continue
    try:
        return int((pattern.get("features", {}) or {}).get("count", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def closed_keep(items, mode="closed"):
    """Decide which candidate sequences survive pruning.

    ``items``: list of ``(seq_tuple, count)``. Returns a list of bools
    (same order): True = keep. Processing is longest-first, then
    highest-count, then lexical — fully deterministic.

    - ``mode="closed"``: drop a candidate when a kept LONGER (or equal,
      earlier) sequence contains it with the SAME count.
    - ``mode="maximal"``: drop it when a kept longer sequence contains
      it, regardless of count.
    """
    if mode not in ("closed", "maximal"):
        raise ValueError("mode must be 'closed' or 'maximal', got %r" % (mode,))
    order = sorted(range(len(items)),
                   key=lambda i: (-len(items[i][0]), -items[i][1], str(items[i][0]), i))
    keep = [True] * len(items)
    kept_seq = []  # (seq, count) of survivors, longest-first
    for i in order:
        seq, cnt = items[i][0], items[i][1]
        drop = False
        for kseq, kcnt in kept_seq:
            if not is_contiguous_subsequence(seq, kseq):
                continue
            if len(kseq) == len(seq) and kseq != seq:
                continue  # same length but different content: unrelated
            if mode == "maximal" or kcnt == cnt:
                drop = True
                break
        if drop:
            keep[i] = False
        else:
            kept_seq.append((seq, cnt))
    return keep


def prune_patterns(patterns, mode="closed"):
    """Split pattern dicts into ``(kept, dropped)`` by closed/maximal rule.

    Returns two new lists; inputs are never mutated. ``dropped`` entries
    are ``(pattern, reason)`` with a human-readable reason citing the
    subsuming sequence and the support count.
    """
    items = [(norm_key(p.get("sequence", [])), count_of(p)) for p in patterns]
    keep = closed_keep(items, mode=mode)
    kept, dropped = [], []
    kept_keys = [(items[i][0], items[i][1]) for i in range(len(patterns)) if keep[i]]
    for i, p in enumerate(patterns):
        if keep[i]:
            kept.append(p)
            continue
        reason = "no surviving super-pattern"
        for kseq, kcnt in kept_keys:
            if (is_contiguous_subsequence(items[i][0], kseq)
                    and (len(kseq) > len(items[i][0]) or kseq == items[i][0])
                    and (mode == "maximal" or kcnt == items[i][1])):
                reason = ("%s subsumed by %s with %s support %d"
                          % (list(items[i][0]), list(kseq),
                             "equal" if mode == "closed" else "a surviving",
                             items[i][1]))
                break
        dropped.append((p, reason))
    return kept, dropped
