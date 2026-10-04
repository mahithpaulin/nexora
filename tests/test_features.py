"""Feature / discovery / matching tests on synthetic ground truth."""
import math

import pytest

from nexora.features.statistical import describe_series
from nexora.features.temporal import (autocorrelation, detect_trend,
                                      fft_periodicity, moving_average)
from nexora.features.sequence import (ngrams, run_length_encode,
                                      transition_counts, transition_probs)
from nexora.features.structural import (association_rules,
                                        connected_components,
                                        cooccurrence_graph)
from nexora.discovery.frequency import find_distribution_modes, find_recurring_values
from nexora.discovery.sequences import find_frequent_sequences
from nexora.discovery.change_points import (change_points, detect_runs,
                                            label_change_points)
from nexora.matching.distance import euclidean, manhattan, normalized_similarity
from nexora.matching.similarity import (cosine_similarity, label_overlap,
                                        pearson_similarity,
                                        sequence_similarity)
from nexora.matching.dtw import dtw_distance


def test_describe_series_known_values():
    d = describe_series([1, 2, 3, 4, 5])
    assert d["mean"] == pytest.approx(3.0)
    assert d["median"] == pytest.approx(3.0)
    assert d["min"] == 1.0 and d["max"] == 5.0
    assert d["n"] == 5 and d["missing"] == 0


def test_describe_series_missing_and_constant():
    d = describe_series([2.0, None, 2.0, float("nan"), 2.0])
    assert d["missing"] == 2 and d["stdev"] == 0.0
    assert all(z == 0.0 for z in d["zscores"] if z is not None)
    d = describe_series([None, None])
    assert d["n"] == 0 and d["mean"] is None


def test_moving_average_and_trend():
    assert moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]
    t = detect_trend([1, 2, 3, 4, 5])
    assert t["direction"] == "rising" and t["slope"] > 0
    t = detect_trend([5, 4, 3, 2, 1])
    assert t["direction"] == "falling"


def test_autocorrelation_periodic():
    ac = autocorrelation([1, -1, 1, -1, 1, -1, 1, -1], max_lag=4)
    assert ac[2] > 0.5  # even lags align for period-2 signal


def test_fft_finds_period():
    sig = [math.sin(2 * math.pi * i / 8) for i in range(64)]
    f = fft_periodicity(sig)
    assert f["method"] == "dft-naive"
    assert any(abs(p - 8) < 1.0 for p in f["periods"])


def test_ngrams_and_transitions():
    assert ngrams(["A", "B", "C", "A"], 2) == [("A", "B"), ("B", "C"), ("C", "A")]
    assert transition_counts("ABCA") == {("A", "B"): 1, ("B", "C"): 1, ("C", "A"): 1}
    probs = transition_probs(["A", "B", "A", "C"])
    assert probs[("A", "B")] == pytest.approx(0.5)
    assert run_length_encode([1, 1, 2]) == [(1, 2), (2, 1)]


def test_structural_graph_and_rules():
    g = cooccurrence_graph(["a", "b", "a", "c"], window=2)
    assert "a" in g["nodes"]
    comps = connected_components(g)
    assert sum(len(c) for c in comps) == len(g["nodes"])
    rules = association_rules([["a", "b"], ["a", "b"], ["a"]], min_support=2)
    assert any(r["antecedent"] == "a" for r in rules)


def test_recurring_values_ground_truth():
    data = list("ABCABCABC")
    pats = find_recurring_values(data, min_support=2)
    by_val = {p["features"]["value"]: p for p in pats}
    assert by_val["A"]["features"]["count"] == 3
    assert by_val["B"]["occurrences"] == [1, 4, 7]


def test_frequent_sequences_abc():
    data = list("ABCABCABC")
    pats = find_frequent_sequences(data, max_n=3, min_support=2)
    seqs = [tuple(p["sequence"]) for p in pats]
    assert ("A", "B", "C") in seqs
    abc = next(p for p in pats if tuple(p["sequence"]) == ("A", "B", "C"))
    assert abc["features"]["count"] == 3


def test_noisy_pattern_still_found():
    data = list("ABCAXCABC")  # one corrupted repeat
    pats = find_frequent_sequences(data, max_n=2, min_support=2)
    assert any(tuple(p["sequence"]) == ("A", "B") for p in pats)


def test_random_data_no_strong_pattern():
    import random
    rng = random.Random(42)
    data = [str(rng.randint(0, 99)) for _ in range(60)]
    pats = find_frequent_sequences(data, max_n=3, min_support=5)
    assert pats == []  # sparse random tokens: nothing frequent


def test_change_points_step():
    vals = [1.0] * 20 + [10.0] * 20
    cps = change_points(vals, window=5, threshold_z=2.0)
    assert any(15 <= c["index"] <= 25 for c in cps)


def test_label_changes_and_runs():
    assert label_change_points(["a", "a", "b"]) == [
        {"index": 2, "before": "a", "after": "b",
         "reason": "label 'a' -> 'b' at 2"}]
    assert detect_runs(["a", "a", "b"]) == [
        {"value": "a", "start": 0, "end": 1, "length": 2},
        {"value": "b", "start": 2, "end": 2, "length": 1}]


def test_distances_and_similarities():
    assert euclidean([1, 2], [1, 2]) == 0.0
    assert manhattan([1, 2], [4, 6]) == pytest.approx(7.0)
    assert normalized_similarity(0.0) == 1.0
    s, _ = cosine_similarity([1, 0], [1, 0])
    assert s == pytest.approx(1.0)
    s, _ = pearson_similarity([1, 2, 3], [1, 2, 3])
    assert s == pytest.approx(1.0)
    s, _ = sequence_similarity(list("ABC"), list("ABC"))
    assert s == pytest.approx(1.0)
    s, _ = label_overlap(["a", "b"], ["b", "c"])
    assert s == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        euclidean([1], [1, 2])


def test_dtw_identical_and_shifted():
    dist, path, sim = dtw_distance([1, 2, 3], [1, 2, 3])
    assert dist == 0.0 and sim == 1.0
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    dist2, _, sim2 = dtw_distance([1, 2, 3], [1, 1, 2, 3])
    assert dist2 >= 0.0 and 0.0 < sim2 <= 1.0
