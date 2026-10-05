"""Workstream 3 (engine wiring): significance filtering in discover().

Null baseline = deterministic seeded shuffles (+ analytic expectation).
Sequential patterns that could easily occur by chance are dropped;
singletons (factual counts) and tiny inputs (n < sig_min_n) never are.
Documented bar: on seeded uniform noise, ZERO sequential patterns
survive (false-pattern rate 0.0 < 5% threshold).
"""
import random

from nexora import Nexora
from nexora.discovery.significance import annotate, annotation_stats


def _sig(p):
    return p.get("significance", {}) or {}


def test_planted_pattern_found_with_numbers():
    out = Nexora().discover(list("ABC") * 10)
    abc = [p for p in out["patterns"] if p["sequence"] == ["A", "B", "C"]]
    assert len(abc) == 1
    feats = _sig(abc[0])
    assert feats["p_value"] < 0.05
    assert feats["lift"] > 1.5
    assert feats["expected_support"] < abc[0]["frequency"]
    sig = out["evidence"]["significance"]
    assert sig["annotated"] >= 1 and sig["skipped"] is None
    assert sig["thresholds"]["alpha_effective"] == 0.05 / 3


def test_tiny_input_annotated_never_dropped():
    out = Nexora().discover(list("ABCABCABC"))
    assert any(p["sequence"] == ["A", "B", "C"] for p in out["patterns"])
    sig = out["evidence"]["significance"]
    assert sig["dropped"] == 0
    assert "sig_min_n" in (sig["skipped"] or "")


def test_noise_sequential_false_pattern_rate_zero():
    rng = random.Random(5)
    noise = [str(rng.randint(0, 19)) for _ in range(300)]
    out = Nexora().discover(noise)
    seqs = [p for p in out["patterns"] if len(p["sequence"]) >= 2]
    assert seqs == []
    sig = out["evidence"]["significance"]
    assert sig["dropped"] >= 20  # the chance bigrams/trigrams, all caught
    rate = len(seqs) / max(1, len(out["patterns"]))
    assert rate == 0.0  # documented bar: < 5% sequential false patterns


def test_singletons_are_factual_never_dropped():
    rng = random.Random(5)
    noise = [str(rng.randint(0, 19)) for _ in range(300)]
    out = Nexora().discover(noise)
    singles = [p for p in out["patterns"] if len(p["sequence"]) == 1]
    assert len(singles) == 20  # all 20 tokens genuinely recurred ~15x
    assert all(p["frequency"] >= 3 for p in singles)


def test_show_all_and_opt_out_store_unvetted():
    data = [str(random.Random(5).randint(0, 19)) for _ in range(120)]
    out = Nexora().discover(data, show_all=True)
    assert out["evidence"]["significance"]["skipped"] == "show_all=True: stored unvetted."
    assert any(len(p["sequence"]) >= 2 for p in out["patterns"])
    out2 = Nexora(config={"significance": False}).discover(data)
    assert all("p_value" not in p["features"] for p in out2["patterns"])


def test_significance_deterministic():
    data = list("ABC") * 12
    a = Nexora().discover(data)["patterns"]
    b = Nexora().discover(data)["patterns"]
    pa = sorted(_sig(p)["p_value"] for p in a)
    pb = sorted(_sig(p)["p_value"] for p in b)
    assert pa == pb


def test_batch_stats_agree_with_annotate():
    data = list("ABC") * 10
    nx = Nexora()
    cands = nx.discover(data, show_all=True)["patterns"]
    cands = [p for p in cands if p["type"] in ("recurring_value", "frequent_sequence")]
    labels = [str(v) for v in data]
    r1 = annotate(cands, labels, n_shuffles=99, seed=42)
    r2 = annotation_stats(cands, labels, n_shuffles=99, seed=42)
    assert [p["sequence"] for p in r1] == [p["sequence"] for p in r2]
    for a, b in zip(r1, r2):
        assert a["expected_support"] == b["expected_support"]
        assert a["lift"] == b["lift"]
        # Same verdict on planted structure with either shuffle stream.
        assert (a["p_value"] < 0.05) == (b["p_value"] < 0.05)
    abc1 = next(p for p in r1 if p["sequence"] == ["A", "B", "C"])
    assert abc1["p_value"] < 0.05 and abc1["lift"] > 1.5
