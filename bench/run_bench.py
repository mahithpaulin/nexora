"""Evaluation harness runner: ``python bench/run_bench.py`` from the repo root.

Uses ONLY the public API (``from nexora import Nexora``) plus the sibling
``bench`` modules. Computes:

* discovery precision/recall/F1 on planted periodic data (found
  frequent-sequence bigrams vs the planted cycle bigrams) and discovery F1
  on pure noise (no structure planted, so any found pattern is spurious);
* anomaly F1 on planted spikes with a +-1 index tolerance;
* next-token accuracy on the held-out tail of the periodic series, alongside
  a naive train-mode baseline for reference.

Prints an ASCII table with columns dataset/metric/score, checks every score
against ``bench/floors.py``, and exits nonzero if any metric is below floor.
"""

from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Tuple

from nexora import Nexora

from bench.datasets import noise as make_noise
from bench.datasets import periodic as make_periodic
from bench.datasets import planted_anomalies as make_spikes
from bench.floors import FLOORS
from bench.metrics import precision_recall_f1

TOL = 1
PERIOD = ["A", "B", "C"]
TRAIN_N = 250


def _expected_bigrams(period: List[str]) -> set:
    """Consecutive pairs of the cycle, including the wrap-around pair."""
    return {
        "%s>%s" % (a, b) for a, b in zip(period, period[1:] + period[:1])
    }


def _found_bigrams(patterns: List[Dict[str, Any]]) -> set:
    """Bigram strings extracted from discovered frequent-sequence patterns."""
    got = set()
    for p in patterns:
        if not isinstance(p, dict) or p.get("type") != "frequent_sequence":
            continue
        seq = [str(x) for x in (p.get("sequence") or [])]
        for a, b in zip(seq, seq[1:]):
            got.add("%s>%s" % (a, b))
    return got


def eval_discovery(data: List[Any], expected: set) -> Tuple[Dict[str, float], int]:
    """Discover patterns via the public API; score found vs expected bigrams."""
    nx = Nexora()
    res = nx.discover(data)
    got = _found_bigrams(res.get("patterns", []))
    return precision_recall_f1(expected, got), len(got)


def eval_anomalies(
    data: List[float], expected_indices: List[int], tol: int = TOL
) -> Tuple[Dict[str, float], List[int]]:
    """Anomaly PRF with greedy +-``tol`` matching of predicted to true indices."""
    nx = Nexora()
    res = nx.find_anomalies(data)
    got = sorted(
        {
            int(a["index"])
            for a in res.get("anomalies", [])
            if isinstance(a, dict) and "index" in a
        }
    )
    remaining = list(got)
    tp = 0
    for e in sorted(expected_indices):
        best = None
        for g in remaining:
            if abs(g - e) <= tol and (
                best is None or abs(g - e) < abs(best - e)
            ):
                best = g
        if best is not None:
            tp += 1
            remaining.remove(best)
    fp = len(got) - tp
    fn = len(expected_indices) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return (
        {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        got,
    )


def eval_prediction(
    data: List[str], train_n: int = TRAIN_N
) -> Tuple[float, float, str]:
    """Held-out tail accuracy of ``predict_next`` vs a naive baseline.

    The baseline always predicts the most frequent training token (order-0);
    the engine uses its Markov/context model (order >= 1), hence the note.
    """
    train, test = data[:train_n], data[train_n:]
    nx = Nexora()
    hits = 0
    for k in range(len(test)):
        hist = data[: train_n + k]
        try:
            nxt = nx.predict_next(hist, current=hist[-1]).get("next")
        except Exception:
            nxt = None
        if str(nxt) == str(test[k]):
            hits += 1
    acc = hits / len(test) if test else 0.0
    mode = collections.Counter(str(x) for x in train).most_common(1)[0][0]
    base = sum(1 for x in test if str(x) == mode) / len(test) if test else 0.0
    note = (
        "order-1 engine (Markov/context predict_next, acc %.3f) vs "
        "order-0 baseline (always train mode %r, acc %.3f)" % (acc, mode, base)
    )
    return acc, base, note


def run_all() -> Tuple[List[Tuple[str, str, float]], List[str]]:
    """Run every evaluation; return ([(dataset, metric, score)], notes)."""
    rows: List[Tuple[str, str, float]] = []
    notes: List[str] = []

    pdata, _ = make_periodic()
    prf, n_found = eval_discovery(pdata, _expected_bigrams(PERIOD))
    rows.append(("periodic", "discovery_precision", prf["precision"]))
    rows.append(("periodic", "discovery_recall", prf["recall"]))
    rows.append(("periodic", "discovery_f1", prf["f1"]))
    notes.append(
        "periodic discovery: %d frequent-sequence bigrams found" % n_found
    )

    ndata, _ = make_noise()
    nprf, n_noise = eval_discovery(ndata, set())
    rows.append(("noise", "discovery_f1", nprf["f1"]))
    notes.append(
        "noise discovery: %d spurious frequent-sequence bigrams "
        "(expected none)" % n_noise
    )

    sdata, struth = make_spikes()
    aprf, got_idx = eval_anomalies(sdata, list(struth["anomaly_indices"]))
    rows.append(("planted_anomalies", "anomaly_f1", aprf["f1"]))
    notes.append(
        "anomaly tolerance +-%d: tp=%d fp=%d fn=%d" % (TOL, aprf["tp"], aprf["fp"], aprf["fn"])
    )

    acc, base, note = eval_prediction(pdata)
    rows.append(("periodic", "predict_accuracy", acc))
    rows.append(("periodic", "baseline_accuracy", base))
    notes.append("prediction note: " + note)

    return rows, notes


def render_table(rows: List[Tuple[str, str, float]]) -> str:
    """ASCII table with columns dataset/metric/score."""
    w_d = max(len("dataset"), *(len(d) for d, _, _ in rows))
    w_m = max(len("metric"), *(len(m) for _, m, _ in rows))
    lines = [
        "%-*s | %-*s | %s" % (w_d, "dataset", w_m, "metric", "score"),
        "%s-+-%s-+-%s" % ("-" * w_d, "-" * w_m, "-" * 5),
    ]
    for d, m, s in rows:
        lines.append("%-*s | %-*s | %.3f" % (w_d, d, w_m, m, s))
    return "\n".join(lines)


def main() -> int:
    """Entry point: print table, enforce floors, return process exit code."""
    rows, notes = run_all()
    print(render_table(rows))
    for note in notes:
        print("note: " + note)
    failures = []
    for d, m, s in rows:
        key = "%s/%s" % (d, m)
        floor = float(FLOORS.get(key, 0.0))
        if s < floor:
            failures.append("%s score %.3f < floor %.3f" % (key, s, floor))
    if failures:
        for msg in failures:
            print("FAIL: " + msg, file=sys.stderr)
        return 1
    print("PASS: all %d metrics >= floors." % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
