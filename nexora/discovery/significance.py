"""Statistical significance for discovered sequential patterns (stdlib only).

Null model: tokens are independent draws with marginal probabilities estimated
from ``labels``.  All helpers are deterministic — the permutation test uses a
local ``random.Random(seed)`` instance and never touches global state.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence


def _count_occurrences(labels: Sequence[str], pattern_seq: Sequence[str]) -> int:
    """Count contiguous occurrences of ``pattern_seq`` in ``labels``.

    Length-1 sequences count exact matches; longer ones count (possibly
    overlapping) contiguous windows equal to ``pattern_seq``.
    """
    seq = list(pattern_seq)
    if not seq:
        return 0
    if len(seq) == 1:
        token = seq[0]
        return sum(1 for x in labels if x == token)
    k = len(seq)
    return sum(
        1 for i in range(len(labels) - k + 1) if list(labels[i : i + k]) == seq
    )


def expected_support(labels: Sequence[str], pattern_seq: Sequence[str]) -> float:
    """Analytic null expectation under token independence.

    ``n_slots * product(marginal prob of each token)`` where ``n_slots`` is
    ``len(labels) - len(seq) + 1`` for sequences (contiguous occurrences) or
    ``len(labels)`` for single values.  Returns 0.0 if a token never appears
    (or on empty input / pattern longer than ``labels``).
    """
    seq = list(pattern_seq)
    n = len(labels)
    if n == 0 or not seq:
        return 0.0
    if len(seq) == 1:
        n_slots = n
    else:
        n_slots = n - len(seq) + 1
    if n_slots <= 0:
        return 0.0
    product = 1.0
    for token in seq:
        count = sum(1 for x in labels if x == token)
        if count == 0:
            return 0.0
        product *= count / n
    return n_slots * product


def pattern_lift(support: float, expected: float) -> float:
    """Observed-over-expected ratio, floored to avoid division by zero."""
    return support / max(expected, 1e-9)


def shuffle_p_value(
    labels: Sequence[str],
    pattern_seq: Sequence[str],
    observed_support: int,
    n_shuffles: int = 199,
    seed: int = 42,
) -> float:
    """Deterministic permutation test p-value.

    Shuffles a copy of ``labels`` with ``random.Random(seed)``, counts
    contiguous occurrences of ``pattern_seq`` in each shuffle, and returns
    ``(1 + n_ge) / (1 + n_shuffles)`` where ``n_ge`` is the number of
    shuffles with count >= ``observed_support``.
    """
    seq = list(pattern_seq)
    if not seq or len(labels) == 0 or n_shuffles <= 0:
        return 1.0
    rng = random.Random(seed)
    base = list(labels)
    n_ge = 0
    for _ in range(n_shuffles):
        shuffled = base[:]
        rng.shuffle(shuffled)
        if _count_occurrences(shuffled, seq) >= observed_support:
            n_ge += 1
    return (1 + n_ge) / (1 + n_shuffles)


def annotate(
    patterns: list[dict],
    labels: Sequence[str],
    n_shuffles: int = 99,
    seed: int = 42,
    max_patterns: int = 200,
) -> list[dict]:
    """Return NEW pattern dicts annotated with significance statistics.

    Each input pattern must carry a ``"sequence"`` list and a ``"frequency"``
    or ``"support"`` int.  Adds ``"expected_support"``, ``"lift"`` and
    ``"p_value"`` keys without mutating the input.  Patterns with an empty
    sequence are skipped; work is capped at ``max_patterns`` entries.
    """
    annotated: list[dict] = []
    for pattern in patterns[:max_patterns]:
        seq = pattern.get("sequence", [])
        if not seq:
            continue
        seq_list = list(seq)
        if "support" in pattern:
            support = pattern["support"]
        elif "frequency" in pattern:
            support = pattern["frequency"]
        else:
            support = 0
        expected = expected_support(labels, seq_list)
        lift = pattern_lift(support, expected)
        p_value = shuffle_p_value(
            labels, seq_list, support, n_shuffles=n_shuffles, seed=seed
        )
        new_pattern = dict(pattern)
        new_pattern["sequence"] = seq_list
        new_pattern["expected_support"] = expected
        new_pattern["lift"] = lift
        new_pattern["p_value"] = p_value
        annotated.append(new_pattern)
    return annotated


def annotation_stats(
    patterns: list[dict],
    labels: Sequence[str],
    n_shuffles: int = 199,
    seed: int = 42,
    max_patterns: int = 500,
) -> list[dict]:
    """Like :func:`annotate` but all p-values share one shuffle stream.

    One permutation loop scores every candidate at once (grouped by
    length, window scans with set lookups), so cost is O(shuffles * n)
    instead of O(shuffles * n * patterns). Return shape is identical
    to :func:`annotate` (new dicts, inputs never mutated). P-values
    may differ in the last shuffle from :func:`annotate` (one shared
    stream vs one stream per pattern); both are valid permutation
    p-values and both are deterministic for a fixed seed.
    """
    items: list[tuple[list, int]] = []
    for pattern in patterns[:max_patterns]:
        seq = list(pattern.get("sequence", []) or [])
        if not seq:
            continue
        if "support" in pattern:
            support = pattern["support"]
        elif "frequency" in pattern:
            support = pattern["frequency"]
        else:
            support = 0
        try:
            support = int(support)
        except (TypeError, ValueError):
            support = 0
        items.append((seq, support))
    labs = list(labels)
    nsh = max(0, int(n_shuffles))
    by_len: dict[int, dict[tuple, list[int]]] = {}
    for i, (seq, _sup) in enumerate(items):
        by_len.setdefault(len(seq), {}).setdefault(tuple(seq), []).append(i)
    ge = [0] * len(items)
    if nsh > 0 and labs:
        rng = random.Random(seed)
        for _ in range(nsh):
            sh = labs[:]
            rng.shuffle(sh)
            for length, table in by_len.items():
                counts: dict = {}
                if length == 1:
                    for x in sh:
                        counts[x] = counts.get(x, 0) + 1
                    for tup, idxs in table.items():
                        try:
                            cc = counts.get(tup[0], 0)
                        except TypeError:
                            cc = 0
                        for i in idxs:
                            if cc >= items[i][1]:
                                ge[i] += 1
                else:
                    if len(sh) >= length:
                        for j in range(len(sh) - length + 1):
                            try:
                                w = tuple(sh[j:j + length])
                            except TypeError:
                                continue
                            if w in table:
                                counts[w] = counts.get(w, 0) + 1
                    for tup, idxs in table.items():
                        cc = counts.get(tup, 0)
                        for i in idxs:
                            if cc >= items[i][1]:
                                ge[i] += 1
    out: list[dict] = []
    kept = 0
    for pattern in patterns[:max_patterns]:
        seq = list(pattern.get("sequence", []) or [])
        if not seq:
            continue
        if "support" in pattern:
            support = pattern["support"]
        elif "frequency" in pattern:
            support = pattern["frequency"]
        else:
            support = 0
        expected = expected_support(labs, seq)
        lift = pattern_lift(support, expected)
        p_value = (1 + ge[kept]) / (1 + nsh) if nsh > 0 and labs else 1.0
        kept += 1
        new_pattern = dict(pattern)
        new_pattern["sequence"] = seq
        new_pattern["expected_support"] = expected
        new_pattern["lift"] = lift
        new_pattern["p_value"] = p_value
        out.append(new_pattern)
    return out


def is_significant(
    p_value: float,
    lift: float,
    support: int,
    min_support: int = 3,
    p_threshold: float = 0.05,
    min_lift: float = 1.5,
) -> tuple[bool, str]:
    """Significance decision with a human-readable reason citing numbers."""
    if support < min_support:
        return (
            False,
            f"not significant: support {support} < min_support {min_support}",
        )
    if p_value >= p_threshold:
        return (
            False,
            f"not significant: p_value {p_value:.4g} >= p_threshold "
            f"{p_threshold} (lift {lift:.4g}, support {support})",
        )
    if lift < min_lift:
        return (
            False,
            f"not significant: lift {lift:.4g} < min_lift {min_lift} "
            f"(p_value {p_value:.4g}, support {support})",
        )
    return (
        True,
        f"significant: support {support} >= {min_support}, "
        f"p_value {p_value:.4g} < {p_threshold}, lift {lift:.4g} >= {min_lift}",
    )
