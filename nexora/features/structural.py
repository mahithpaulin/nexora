"""Structural features: co-occurrence graphs, components, association rules.

Stdlib only. Works on categorical label sequences or on lists of
item-sets (transactions). All scores are алифатические counts / ratios
in 0..1 with documented meaning; missing (None) tokens are skipped.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations


def cooccurrence_graph(labels, window=2):
    """Undirected co-occurrence graph over labels within a sliding window.

    Returns {"nodes": [...], "edges": {(a, b): count}} with a < b
    canonical ordering. ``window`` is the number of consecutive items
    that count as co-occurring (>= 2).
    """
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an int >= 2")
    items = [t for t in list(labels) if t is not None]
    edges: dict[tuple, int] = {}
    for i in range(len(items)):
        for j in range(i + 1, min(i + window, len(items))):
            a, b = items[i], items[j]
            if a == b:
                continue
            key = (a, b) if str(a) <= str(b) else (b, a)
            edges[key] = edges.get(key, 0) + 1
    nodes = sorted(set(items), key=str)
    return {"nodes": nodes, "edges": edges}


def connected_components(graph):
    """Connected components of an undirected graph (BFS, deterministic).

    ``graph``: {"nodes": [...], "edges": {(a,b): count}}.
    Returns a list of sorted node lists, ordered by first node.
    """
    nodes = list(graph.get("nodes", []))
    edges = graph.get("edges", {})
    adj = defaultdict(set)
    for (a, b) in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    comps = []
    for start in sorted(nodes, key=str):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in sorted(adj[cur], key=str):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(sorted(comp, key=str))
    comps.sort(key=lambda c: str(c[0]) if c else "")
    return comps


def association_rules(transactions, min_support=2, min_confidence=0.5):
    """Single-antecedent association rules A -> C from item-set transactions.

    support = P(A and C) as a count fraction over transactions (0..1);
    confidence = P(C|A) (0..1). Returns list of
    {"antecedent", "consequent", "support", "confidence", "count"},
    sorted by (-confidence, -support, str(rule)). Deterministic.
    """
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in 0..1")
    txns = [set(t) for t in (transactions or []) if t]
    n = len(txns)
    if n == 0:
        return []
    item_count = Counter()
    pair_count = Counter()
    for t in txns:
        for item in t:
            item_count[item] += 1
        for a, c in combinations(sorted(t, key=str), 2):
            pair_count[(a, c)] += 1
    rules = []
    for (a, c), cnt in pair_count.items():
        if cnt < min_support:
            continue
        for ant, con in ((a, c), (c, a)):
            conf = cnt / item_count[ant] if item_count[ant] else 0.0
            if conf >= min_confidence:
                rules.append({
                    "antecedent": ant, "consequent": con,
                    "support": cnt / n, "confidence": conf, "count": cnt,
                })
    rules.sort(key=lambda r: (-r["confidence"], -r["support"],
                              str(r["antecedent"]), str(r["consequent"])))
    return rules
