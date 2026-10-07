"""Loop 2 batch N (I71-I75)."""
import pytest

from nexora.api.engine import Nexora


def test_validate():
    nx = Nexora()
    v = nx.validate([1.0, None, "x"])
    assert v["status"] == "FOUND" and v["junk"] == 1 and v["n"] == 3
    assert nx.validate([])["status"] == "INSUFFICIENT_DATA"


def test_dedupe():
    nx = Nexora()
    d = nx.dedupe(["A", "A", "B", "B", "B", "A"])
    assert d["data"] == ["A", "B", "A"] and d["dropped"] == 3


def test_clip():
    nx = Nexora()
    c = nx.clip([-5.0, 0.0, 5.0], lo=0.0, hi=1.0)
    assert c["data"] == [0.0, 0.0, 1.0] and c["clipped"] == 2
    with pytest.raises(ValueError):
        nx.clip([1.0], lo=2.0, hi=1.0)


def test_fill_missing():
    nx = Nexora()
    f = nx.fill_missing([1.0, None, None, 4.0])
    assert f["data"] == [1.0, 1.0, 1.0, 4.0] and f["filled"] == 2
    m = nx.fill_missing([1.0, None, 3.0], method="mean")
    assert m["data"] == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        nx.fill_missing([1.0], method="zero")


def test_outliers_iqr():
    nx = Nexora()
    o = nx.outliers_iqr([10.0] * 10 + [100.0])
    assert o["status"] == "FOUND" and o["indices"] == [10]
    assert nx.outliers_iqr([1.0, 2.0, 3.0, 4.0])["status"] == "NONE"
    assert nx.outliers_iqr([1.0])["status"] == "INSUFFICIENT_DATA"
