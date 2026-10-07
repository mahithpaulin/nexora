"""Workstream 7: true incremental streaming on the engine.

update() folds observations in bounded memory (no history
reprocessing); stream_predict() matches batch predict(); online
change detection fires on regime shifts. 100k-observation test
documents the memory bound actually measured (peak 2.63MB on the
dev box; bound set at 15MB for CI headroom).
"""
import collections
import statistics
import tracemalloc

from nexora import Nexora

MEMORY_BOUND_MB = 15.0


class CountingDeque(collections.deque):
    """Deque that counts iterations (white-box: update must not scan it)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.iters = 0

    def __iter__(self):
        self.iters += 1
        return super().__iter__()


def test_update_stats_match_batch_100k():
    data = [float(i % 100) for i in range(100000)]
    nx = Nexora()
    tracemalloc.start()
    for i in range(0, len(data), 10000):
        nx.update(data[i:i + 10000])
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / 1e6 < MEMORY_BOUND_MB
    s = nx.update([])["stats"]
    assert s["n"] == 100000
    assert abs(s["mean"] - statistics.fmean(data)) < 1e-9
    assert abs(s["stdev"] - statistics.pstdev(data)) / statistics.pstdev(data) < 1e-9


def test_update_predict_matches_batch():
    data = list("ABC") * 33333 + ["A"]
    assert len(data) == 100000
    nx = Nexora()
    for i in range(0, len(data), 10000):
        nx.update(data[i:i + 10000])
    sp = nx.stream_predict()
    batch = Nexora().predict(data)
    assert sp["current"] == "A"
    assert sp["predictions"][0]["next"] == batch["predictions"][0]["next"] == "B"
    assert sp["predictions"][0]["probability"] == batch["predictions"][0]["probability"] == 1.0
    # Bounded structures, not O(n): 1024 recent rows, 3 bigrams, <=100 events.
    assert len(nx._history) == 1024
    assert len(nx._stream_trans) == 3
    assert len(nx._stream_changes) <= 100


def test_no_reprocessing_of_history():
    nx = Nexora()
    nx._history = CountingDeque(maxlen=1024)
    for v in [1.0, 2.0, 3.0]:
        nx._history.append({"index": v, "value": v, "label": str(v)})
    nx.update([10.0] * 2000 + list("ABCDE") * 20)
    assert nx._history.iters == 0
    assert len(nx._history) == 1024  # capped, not 3 + 2100
    assert nx._history[-1]["stream_pos"] == nx._stream_n - 1


def test_chunks_equal_one_shot():
    data = [float(i % 7) for i in range(500)] + list("ABCABC")
    a, b = Nexora(), Nexora()
    a.update(data)
    for i in range(0, len(data), 37):
        b.update(data[i:i + 37])
    sa, sb = a.update([])["stats"], b.update([])["stats"]
    assert sa == sb
    assert a.stream_predict()["predictions"] == b.stream_predict()["predictions"]
    assert a._stream_trans == b._stream_trans


def test_online_change_detection():
    nx = Nexora()
    nx.update([0.0] * 150)
    assert nx.update([])["changes_total"] == 0
    out = nx.update([5.0] * 150)
    assert out["changes"], "sustained 0->5 shift must fire at least once"
    # v3: row index and stream_pos are both global across calls.
    first = min(c["stream_pos"] for c in out["changes"])
    assert 150 <= first <= 170
    assert first == min(c["index"] for c in out["changes"])
    assert all(c["kind"] == "stream_change" for c in out["changes"])
    assert "long-run mean" in out["changes"][0]["explanation"]
    # Bounded: never more than the cap, even on long shifts.
    nx.update([5.0] * 5000)
    assert len(nx._stream_changes) <= 100


def test_update_empty_and_junk():
    nx = Nexora()
    out = nx.update([])
    assert out["processed"] == 0 and out["total"] == 0
    out = nx.update([None, "x", float("nan"), float("inf")])
    assert out["processed"] == 4 and out["total"] == 4
    assert out["stats"]["n"] == 0
    assert nx.stream_predict()["predictions"] == []
    # NOTE: pre-existing ingestion behavior coerces True -> 1.0 before
    # update() ever sees it (bool handling lives in ingestion, not here).
    out = Nexora().update([True])
    assert out["stats"]["n"] == 1


def test_update_config_capacity_and_flags():
    nx = Nexora(config={"stream_capacity": 10})
    nx.update(list(range(50)))
    assert len(nx._history) == 10
    nx2 = Nexora(config={"change_z": 100.0})
    out = nx2.update([0.0] * 150 + [5.0] * 150)
    assert out["changes"] == []  # unreachable threshold: silent, honestly
    try:
        Nexora(config={"stream_capacity": 0})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected fail-fast on stream_capacity=0")
