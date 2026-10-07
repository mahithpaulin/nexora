"""Workstream 8: mining stays correct after the single-pass rewrite.

Outputs (counts, supports, occurrence indices) must be identical to
the old rescan logic; occurrence indices still refer to original
positions when missing values are skipped.
"""
from nexora.discovery.frequency import find_recurring_values
from nexora.discovery.sequences import find_frequent_sequences


def test_positions_skip_missing_without_renumbering():
    labels = ["A", None, "A", "B"]
    seqs = find_frequent_sequences(labels, max_n=2, min_support=1)
    ab = next(p for p in seqs if p["sequence"] == ["A", "B"])
    assert ab["occurrences"] == [2]  # original index, not renumbered
    assert ab["features"]["count"] == 1
    assert ab["features"]["support"] == 1 / 3
    rec = find_recurring_values(labels, min_support=1)
    by_val = {p["sequence"][0]: p for p in rec}
    assert by_val["A"]["occurrences"] == [0, 2]
    assert by_val["B"]["occurrences"] == [3]
    assert by_val["A"]["frequency"] == 2


def test_counts_supports_unchanged_on_clean_data():
    labels = list("ABCABCABC")
    seqs = find_frequent_sequences(labels, max_n=3, min_support=2)
    got = {(tuple(p["sequence"]), p["frequency"]) for p in seqs}
    assert (("A", "B", "C"), 3) in got
    assert (("A", "B"), 3) in got
    assert (("C", "A"), 2) in got
    abc = next(p for p in seqs if p["sequence"] == ["A", "B", "C"])
    assert abc["occurrences"] == [0, 3, 6]
    assert abc["features"]["support"] == 3 / 7


def test_mining_deterministic():
    labels = list("ABACABADABACABA")
    r1 = find_frequent_sequences(labels, max_n=3, min_support=2)
    r2 = find_frequent_sequences(labels, max_n=3, min_support=2)
    assert r1 == r2
    f1 = find_recurring_values(labels, min_support=2)
    f2 = find_recurring_values(labels, min_support=2)
    assert f1 == f2


def test_mining_rejects_bad_args():
    for bad in (lambda: find_frequent_sequences(["A"], max_n=1),
                lambda: find_frequent_sequences(["A"], min_support=0),
                lambda: find_recurring_values(["A"], min_support=0)):
        try:
            bad()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


def test_empty_and_all_missing():
    assert find_frequent_sequences([], max_n=3) == []
    assert find_frequent_sequences([None, None], max_n=2, min_support=1) == []
    assert find_recurring_values([], min_support=1) == []
    assert find_recurring_values([None, float("nan")], min_support=1) == []
