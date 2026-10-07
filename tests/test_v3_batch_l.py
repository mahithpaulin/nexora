"""Loop 2 batch L (I61-I65)."""
import json

import pytest

from nexora.api.engine import Nexora


def test_update_elapsed():
    out = Nexora().update([1.0, 2.0, 3.0])
    assert out["elapsed"] >= 0.0


def test_stream_describe():
    nx = Nexora()
    assert nx.stream_describe()["status"] == "NONE"
    nx.update([1.0, 2.0, 3.0])
    d = nx.stream_describe()
    assert d["status"] == "FOUND" and d["n"] == 3


def test_replay():
    nx = Nexora()
    r = nx.replay([float(i) for i in range(100)], chunk=30)
    assert r["total"] == 100 and len(r["chunks"]) == 4
    assert sum(c["processed"] for c in r["chunks"]) == 100
    with pytest.raises(ValueError):
        nx.replay({"a": 1})


def test_checkpoint_restore():
    a = Nexora()
    a.update([float(i % 5) for i in range(60)])
    a.update([50.0] * 60)
    cp = a.stream_checkpoint()
    json.dumps(cp["state"])  # serializable
    b = Nexora()
    assert b.stream_restore(cp["state"])["n"] == 120
    assert b.stream_stats()["stats"] == a.stream_stats()["stats"]
    assert b.stream_predict()["predictions"] == a.stream_predict()["predictions"]
    assert b.stream_changes()["total"] == a.stream_changes()["total"]
    with pytest.raises(ValueError):
        b.stream_restore({"v": 999})


def test_detrend():
    nx = Nexora()
    d = nx.detrend([1.0, 2.0, 3.0, 4.0], window=2)
    assert d["status"] == "FOUND" and d["residuals"] == [0.0, 0.5, 0.5, 0.5]
    assert nx.detrend([])["status"] == "INSUFFICIENT_DATA"
