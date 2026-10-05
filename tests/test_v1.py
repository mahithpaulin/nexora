"""v1 sweep tests: streaming, perf cache, errors/config, quality, store,
report, batch + engine integration, determinism, perf smoke. Stdlib only.
"""
import json
import math
import statistics
import time

import pytest

from nexora.streaming import (RunningStats, SlidingStats, stream_anomalies,
                              streaming_describe)
from nexora.perf.cache import LRUCache, cached_dtw, memoize, timed
from nexora.matching.dtw import dtw_distance
from nexora.core.errors import (EmptyDataError, InsufficientDataError,
                                InvalidConfigError, NexoraError)
from nexora.core.config import DEFAULTS, validate_config
from nexora.ingestion.quality import quality_report
from nexora.memory.store import load_engine_state, save_engine_state
from nexora.explanation.report import (pattern_card, render_markdown,
                                       render_text)
from nexora.api.batch import (batch_process, compare_signatures,
                              summarize_batch)
from nexora.api.engine import Nexora


# ---------- streaming ----------

def test_running_stats_matches_batch():
    xs = [1.0, 2.0, 3.0, 4.0, None, 5.0]
    rs = RunningStats()
    for x in xs:
        rs.update(x)
    clean = [1, 2, 3, 4, 5]
    assert rs.n == 5 and rs.missing == 1
    assert rs.mean == pytest.approx(statistics.mean(clean))
    assert rs.variance == pytest.approx(statistics.pvariance(clean))
    assert rs.stdev == pytest.approx(statistics.pstdev(clean))
    empty = RunningStats()
    assert empty.mean is None and empty.variance is None


def test_running_stats_merge():
    a, b = RunningStats(), RunningStats()
    for x in [1.0, 2.0, 3.0]:
        a.update(x)
    for x in [4.0, 5.0, 6.0]:
        b.update(x)
    a.merge(b)
    assert a.n == 6
    assert a.mean == pytest.approx(3.5)
    assert a.variance == pytest.approx(statistics.pvariance([1, 2, 3, 4, 5, 6]))
    with pytest.raises(TypeError):
        a.merge("nope")


def test_sliding_window_eviction():
    s = SlidingStats(3)
    for x in [1.0, 2.0, 3.0, 4.0]:
        s.update(x)
    assert s.mean == pytest.approx(3.0)  # [2,3,4]
    assert s.n == 3
    s.update(None)
    assert s.n == 2  # None occupies a slot, excluded from sums
    assert s.mean == pytest.approx(3.5)  # [3,4]
    with pytest.raises(ValueError):
        SlidingStats(0)


def test_stream_anomalies_causal_spike():
    base = [round(10 + (i % 5) * 0.2, 1) for i in range(50)]
    data = base + [40.0] + base[:10]
    out = stream_anomalies(data, window=20, z_threshold=3.0)
    assert [a["index"] for a in out] == [50]
    assert out[0]["score"] >= 0.5
    # warmup suppresses everything early
    assert stream_anomalies([10.0] * 5 + [40.0], window=20) == []


def test_streaming_describe_keys():
    d = streaming_describe([1.0, 2.0, None, 3.0], window=2)
    assert d["n"] == 3 and d["missing"] == 1
    assert d["recent_mean"] == pytest.approx(3.0)  # last window [None,3]


# ---------- perf cache ----------

def test_lru_eviction_and_stats():
    c = LRUCache(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1
    c.set("c", 3)  # evicts b (a was refreshed)
    assert "b" not in c and c.get("a") == 1
    assert c.get("zz") is None  # a miss
    s = c.stats()
    assert s["hits"] == 2 and s["misses"] == 1 and s["size"] == 2
    with pytest.raises(ValueError):
        LRUCache(maxsize=0)


def test_memoize_caches():
    calls = []

    @memoize(maxsize=8)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6 and f(3) == 6 and len(calls) == 1
    assert f.cache_stats()["hits"] == 1
    f.cache_clear()
    assert f.cache_stats()["hits"] == 0


def test_cached_dtw_matches_and_hits():
    cached_dtw.cache_clear()
    d1, s1 = cached_dtw([1, 2, 3], [1, 2, 4])
    raw_d, _, raw_s = dtw_distance([1, 2, 3], [1, 2, 4])
    assert d1 == pytest.approx(raw_d) and s1 == pytest.approx(raw_s)
    cached_dtw([1, 2, 3], [1, 2, 4])
    assert cached_dtw.cache_stats()["hits"] == 1
    assert 0.0 <= s1 <= 1.0


def test_timed_helper():
    (res, secs) = timed(lambda: 42)
    assert res == 42 and secs >= 0.0


# ---------- errors + config ----------

def test_error_hierarchy():
    assert issubclass(EmptyDataError, ValueError)
    assert issubclass(InvalidConfigError, NexoraError)
    e = InsufficientDataError("need more", needed=10, got=3)
    assert (e.needed, e.got) == (10, 3)


def test_validate_config_roundtrip():
    cfg = validate_config(None)
    assert cfg["z_threshold"] == 3.0 and cfg["mv_threshold"] == 0.8
    cfg = validate_config({"z_threshold": 2, "min_support": 3.0, "regimes": False})
    assert cfg["z_threshold"] == 2.0 and cfg["min_support"] == 3
    assert cfg["regimes"] is False
    cfg = validate_config({"bogus_key": 1})
    assert "bogus_key" not in cfg  # unknown keys ignored
    for bad in ({"z_threshold": -5}, {"z_threshold": True},
                {"min_support": 2.5}, {"corr_threshold": 1.5},
                {"regimes": 1}, {"weights": []}):
        with pytest.raises(InvalidConfigError) as ei:
            validate_config(bad)
        assert ei.value.key is not None
    with pytest.raises(InvalidConfigError):
        validate_config("not-a-dict")


# ---------- quality ----------

def test_quality_scores():
    assert quality_report([])["quality"] == 0.0
    assert quality_report([1, 2, 3, 4, 5])["quality"] == pytest.approx(1.0)
    assert quality_report([7, 7, 7])["quality"] == pytest.approx(0.7)  # constant -0.3
    q = quality_report([1, 2, None, None])
    assert q["missing"] == 2 and q["quality"] < 1.0
    assert quality_report(None)["n"] == 0
    with pytest.raises(TypeError):
        quality_report(123)


# ---------- store ----------

def test_store_roundtrip_and_migration(tmp_path):
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABC"))
    f = str(tmp_path / "state.json")
    nx.save(f)
    with open(f) as fh:
        payload = json.load(fh)
    assert payload["version"] == 1
    nx2 = Nexora()
    info = nx2.load(f)
    assert info["patterns"] == nx.repo.size() > 0
    assert nx2.get_pattern("P-001")["frequency"] == nx.get_pattern("P-001")["frequency"]
    # v0 migration
    v0 = str(tmp_path / "v0.json")
    with open(v0, "w") as fh:
        json.dump({"counter": 2, "patterns": nx.repo.all()}, fh)
    mig = load_engine_state(v0)
    assert mig["version"] == 0 and mig["trails"] == {}
    info2 = nx2.load(v0)
    assert info2["version"] == 0
    with pytest.raises(FileNotFoundError):
        load_engine_state(str(tmp_path / "nope.json"))
    fut = str(tmp_path / "fut.json")
    with open(fut, "w") as fh:
        json.dump({"version": 99}, fh)
    with pytest.raises(ValueError):
        load_engine_state(fut)


# ---------- report ----------

def test_report_renders():
    nx = Nexora(config={"min_support": 2})
    res = nx.detect(list("ABCABC"))
    md = nx.report(res)
    assert "# Nexora report" in md and "## Patterns" in md
    assert "P-001" in md
    tx = nx.report(res, fmt="text")
    assert "#" not in tx and "Nexora report" in tx
    assert "P-001" in pattern_card(res["patterns"][0])
    assert isinstance(render_markdown(None), str)
    assert isinstance(render_text({"patterns": "junk"}), str)


# ---------- batch ----------

def test_batch_isolation_and_summary():
    data = [list("ABCABC"), [10.0] * 20 + [50.0]]
    outs = batch_process(data, {"min_support": 2})
    assert len(outs) == 2 and all(o["error"] is None for o in outs)
    s = summarize_batch(outs)
    assert s["n_ok"] == 2 and s["total_patterns"] > 0
    assert s["avg_confidence"] > 0.0
    bad = batch_process([list("ABC"), list("ABC")], {"z_threshold": -5})
    assert all(o["error"] is not None for o in bad)
    assert summarize_batch(bad)["n_failed"] == 2
    assert summarize_batch("junk")["n_runs"] == 0


def test_compare_signatures_diff():
    a = [{"id": "P-1", "type": "t", "features": {"v": 1}, "sequence": [1]}]
    b = a + [{"id": "P-2", "type": "t", "features": {"v": 2}, "sequence": [2]}]
    d = compare_signatures(a, b)
    assert d["added"] == ["P-2"] and d["removed"] == [] and d["common"] == ["P-1"]
    d2 = compare_signatures(a, a)
    assert d2["added"] == d2["removed"] == []


# ---------- engine integration ----------

def test_engine_quality_report_batch_prune():
    nx = Nexora(config={"min_support": 2})
    q = nx.quality([1, 2, None])
    assert q["missing"] == 1 and q["quality"] < 1.0
    outs = nx.batch([list("ABCABC"), list("ABCABC")])
    assert len(outs) == 2
    comp = compare_signatures(outs[0]["result"]["patterns"], outs[1]["result"]["patterns"])
    assert comp["added"] == comp["removed"] == []  # deterministic isolation
    # NOTE (v2/WS2): closed pruning means plain discover() now stores ~1
    # pattern here instead of a dozen redundant ones (D5). show_all=True
    # restores the old unpruned stream so repo.prune() still has >3
    # patterns to trim — which is what this test is actually about.
    nx.discover(list("ABCABC"), show_all=True)
    for p in nx.repo.all():
        nx.repo.update(p["id"], {"state": "RETIRED"})
    before = nx.repo.size()
    removed = nx.repo.prune(max_total=3)
    assert before > 3 and len(removed) == before - 3 and nx.repo.size() == 3
    assert all(nx.repo.get(pid) is None for pid in removed)
    with pytest.raises(InvalidConfigError):
        Nexora({"z_threshold": -1})


def test_determinism_twice_identical():
    data = [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 50.0] + list("ABCABCABC")
    r1 = Nexora().discover(data)["patterns"]
    r2 = Nexora().discover(data)["patterns"]
    d = compare_signatures(r1, r2)
    assert d["added"] == d["removed"] == []
    assert [(p["id"], p["frequency"]) for p in r1] == [(p["id"], p["frequency"]) for p in r2]


def test_perf_smoke_large_series():
    data = [float(i % 7) + (0.1 if i % 2 else 0.0) for i in range(3000)]
    t0 = time.perf_counter()
    disc = Nexora().discover(data)
    dt = time.perf_counter() - t0
    assert disc["patterns"]
    assert dt < 15.0
    t0 = time.perf_counter()
    base = [round(10 + (i % 5) * 0.2, 1) for i in range(19999)]
    out = stream_anomalies(base + [50.0], window=100)
    dt = time.perf_counter() - t0
    assert [a["index"] for a in out] == [19999]
    assert dt < 5.0
