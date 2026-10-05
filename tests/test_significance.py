"""Tests for nexora.discovery.significance (WORKSTREAM 3). Stdlib only."""

import copy
import random

import pytest

from nexora.discovery.significance import (
    annotate,
    expected_support,
    is_significant,
    pattern_lift,
    shuffle_p_value,
)


def _count_windows(labels: list[str], seq: list[str]) -> int:
    k = len(seq)
    return sum(1 for i in range(len(labels) - k + 1) if labels[i : i + k] == seq)


def test_planted_trigram_is_significant() -> None:
    labels = list("ABC") * 10  # "ABC" occurs 10x contiguously
    seq = ["A", "B", "C"]
    observed = _count_windows(labels, seq)
    assert observed == 10
    expected = expected_support(labels, seq)
    assert expected == pytest.approx(28 * (10 / 30) ** 3)
    lift = pattern_lift(observed, expected)
    assert lift > 1.5
    p_value = shuffle_p_value(labels, seq, observed, n_shuffles=99, seed=42)
    assert p_value < 0.05
    annotated = annotate(
        [{"sequence": seq, "frequency": observed}],
        labels,
        n_shuffles=99,
        seed=42,
    )
    assert len(annotated) == 1
    assert annotated[0]["lift"] > 1.5 and annotated[0]["p_value"] < 0.05
    decision, reason = is_significant(
        annotated[0]["p_value"], annotated[0]["lift"], observed
    )
    assert decision is True and "10" in reason


def test_random_trigram_not_significant() -> None:
    rng = random.Random(7)
    alphabet = [f"T{i}" for i in range(20)]
    labels = [rng.choice(alphabet) for _ in range(300)]
    seq = labels[0:3]
    # Sanity: chance data produces no enriched trigram.
    assert _count_windows(labels, seq) <= 2
    # A hypothetical support-3 claim for this chance trigram cannot reach
    # significance on the coarse permutation grid: with n_shuffles=19 the
    # minimum attainable p-value is (1+0)/(1+19) = 0.05, so p >= 0.05 always.
    p_value = shuffle_p_value(labels, seq, 3, n_shuffles=19, seed=123)
    lift = pattern_lift(3, expected_support(labels, seq))
    assert p_value >= 0.05 or lift < 1.5
    decision, _ = is_significant(p_value, lift, 3)
    assert decision is False


def test_p_value_deterministic() -> None:
    labels = list("ABC") * 10
    seq = ["A", "B", "C"]
    first = shuffle_p_value(labels, seq, 10, n_shuffles=49, seed=42)
    second = shuffle_p_value(labels, seq, 10, n_shuffles=49, seed=42)
    assert first == second
    for seed in (1, 2, 3):
        p_value = shuffle_p_value(labels, seq, 10, n_shuffles=49, seed=seed)
        assert 0.0 <= p_value <= 1.0


def test_annotate_never_mutates_input() -> None:
    labels = list("ABC") * 10
    patterns = [
        {"sequence": ["A", "B", "C"], "frequency": 10},
        {"sequence": [], "frequency": 5},
        {"sequence": ["A"], "support": 10},
    ]
    snapshot = copy.deepcopy(patterns)
    annotated = annotate(patterns, labels, n_shuffles=19, seed=42)
    assert patterns == snapshot  # input untouched
    assert all("p_value" not in p for p in patterns)
    # Empty-sequence pattern is skipped; the rest are NEW dicts.
    assert len(annotated) == 2
    assert annotated[0] is not patterns[0]
    assert annotated[0]["sequence"] is not patterns[0]["sequence"]
    assert annotated[0]["sequence"] == ["A", "B", "C"]
    for entry in annotated:
        assert {"expected_support", "lift", "p_value"} <= set(entry)


def test_analytic_expectation_exact() -> None:
    labels = ["A", "A", "B", "A"]
    assert expected_support(labels, ["A"]) == 3.0
    assert expected_support(labels, ["A", "B"]) == 0.5625  # 3 * 0.75 * 0.25
    assert expected_support(labels, ["ZZZ"]) == 0.0
