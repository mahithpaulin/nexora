"""v3 iteration 3: update() rows carry global stream indices.

History row index == stream_pos across chunked calls (no per-call
restart), and change events cite the same global position.
"""
from nexora.api.engine import Nexora


def test_history_indices_global_across_calls():
    nx = Nexora()
    nx.update([0.0, 1.0, 2.0])
    nx.update([3.0, 4.0])
    assert [r["index"] for r in nx._history] == [0, 1, 2, 3, 4]
    assert [r["stream_pos"] for r in nx._history] == [0, 1, 2, 3, 4]


def test_change_event_index_is_global():
    nx = Nexora()
    nx.update([0.0] * 150)
    out = nx.update([5.0] * 150)
    assert out["changes"], "sustained 0->5 shift must fire"
    for c in out["changes"]:
        assert c["index"] == c["stream_pos"] >= 150


def test_chunks_still_equal_one_shot():
    data = [float(i % 7) for i in range(200)]
    a, b = Nexora(), Nexora()
    a.update(data)
    for i in range(0, len(data), 37):
        b.update(data[i:i + 37])
    assert [r["index"] for r in a._history] == [r["index"] for r in b._history]
    assert a.update([])["stats"] == b.update([])["stats"]
