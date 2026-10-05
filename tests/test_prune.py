"""Workstream 2: closed/maximal pattern pruning (D5).

Mining enumerates every frequent n-gram, so 9 observations of ABC
produced 12-15 patterns. Closed pruning drops a sub-pattern when a
longer kept pattern contains it with the same support-count.
"""
import itertools

from nexora import Nexora
from nexora.discovery.prune import (closed_keep, count_of,
                                    is_contiguous_subsequence,
                                    prune_patterns)


def _mk(seq, count):
    return {"sequence": list(seq), "frequency": count}


# ---------- acceptance: ABCABCABC -> ~1-3 patterns ----------

def test_d5_abc_returns_few_patterns():
    out = Nexora().discover(list("ABCABCABC"))
    seqs = [p for p in out["patterns"]
            if p["type"] in ("recurring_value", "frequent_sequence")]
    assert 1 <= len(seqs) <= 3
    assert any(p["sequence"] == ["A", "B", "C"] for p in seqs)


def test_show_all_restores_everything():
    out = Nexora().discover(list("ABCABCABC"), show_all=True)
    seqs = [p for p in out["patterns"]
            if p["type"] in ("recurring_value", "frequent_sequence")]
    assert len(seqs) >= 6  # A,B,C + AB,BC + ABC at min_support=3
    assert out["evidence"]["pruned"]["dropped"] == 0


def test_config_opt_out():
    nx = Nexora(config={"prune_redundant": False})
    out = nx.discover(list("ABCABCABC"))
    seqs = [p for p in out["patterns"]
            if p["type"] in ("recurring_value", "frequent_sequence")]
    assert len(seqs) >= 6


def test_pruned_evidence_cites_numbers():
    out = Nexora().discover(list("ABCABCABC"))
    pr = out["evidence"]["pruned"]
    assert pr["dropped"] >= 3
    assert any("equal support 3" in d for d in pr["details"])


# ---------- closed (not maximal): higher-count sub-patterns survive ----------

def test_higher_count_subpattern_kept():
    # A x3, AA x2: different counts, neither subsumes the other.
    out = Nexora(config={"min_support": 2}).discover(list("AAAB"))
    seqs = {(tuple(p["sequence"]), p["frequency"]) for p in out["patterns"]
            if p["type"] in ("recurring_value", "frequent_sequence")}
    assert (("A",), 3) in seqs
    assert (("A", "A"), 2) in seqs


def test_prune_unit_closed_vs_maximal():
    pats = [_mk("ABC", 3), _mk("AB", 3), _mk("BC", 3),
            _mk("A", 3), _mk("B", 3), _mk("C", 3)]
    kept, dropped = prune_patterns(pats, mode="closed")
    assert [p["sequence"] for p in kept] == [["A", "B", "C"]]
    assert len(dropped) == 5
    assert all("equal support 3" in r for _, r in dropped)
    kept_m, _ = prune_patterns(pats, mode="maximal")
    assert [p["sequence"] for p in kept_m] == [["A", "B", "C"]]
    # Maximal drops even on unequal counts: AB x5 still goes when ABC x3 stays.
    pats2 = [_mk("ABC", 3), _mk("AB", 5)]
    kept2, _ = prune_patterns(pats2, mode="maximal")
    assert [p["sequence"] for p in kept2] == [["A", "B", "C"]]
    kept2c, _ = prune_patterns(pats2, mode="closed")
    assert len(kept2c) == 2


def test_prune_unit_subsequence_edges():
    assert is_contiguous_subsequence(("A",), ("A", "B", "C"))
    assert is_contiguous_subsequence(("B", "C"), ("A", "B", "C"))
    assert not is_contiguous_subsequence(("A", "C"), ("A", "B", "C"))
    assert not is_contiguous_subsequence(("A", "B", "C", "D"), ("A", "B", "C"))
    assert is_contiguous_subsequence(("A", "B"), ("A", "B"))
    assert not is_contiguous_subsequence((), ("A",))
    assert count_of({"frequency": 4}) == 4
    assert count_of({"support": 4}) == 4
    assert count_of({}) == 0


def test_prune_deterministic_and_pure():
    pats = [_mk("ABC", 3), _mk("AB", 3), _mk("A", 3), _mk("BC", 3)]
    before = [dict(p) for p in pats]
    r1 = closed_keep([(("A", "B", "C"), 3), (("A", "B"), 3), (("A",), 3)])
    r2 = closed_keep([(("A", "B", "C"), 3), (("A", "B"), 3), (("A",), 3)])
    assert r1 == r2 == [True, False, False]
    prune_patterns(pats)
    assert pats == before  # inputs never mutated
    # Input order does not change the kept SET.
    out_sets = set()
    for perm in itertools.permutations(pats):
        kept, _ = prune_patterns(list(perm))
        out_sets.add(tuple(sorted(tuple(p["sequence"]) for p in kept)))
    assert out_sets == {(("A", "B", "C"),)}


def test_prune_bad_mode_raises():
    try:
        closed_keep([((("A",), 1))], mode="bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_prune_keeps_association_and_other_types_untouched():
    # Only recurring/sequential candidates enter the prune pool.
    out = Nexora().discover([1.0, 2.0, 3.0] * 6)
    kinds = {p["type"] for p in out["patterns"]}
    assert "frequent_sequence" in kinds or "recurring_value" in kinds
