"""v3 iteration 1: every engine entry point accepts any iterable.

Tuples, ranges and generators are iterated element-wise (same rows as
the equivalent list). Single scalars still wrap as one observation,
and TypeErrors name the fix (wrap as [value]).
"""
import pytest

from nexora.api.engine import Nexora
from nexora.core.observation import normalize_observations
from nexora.ingestion.parser import parse


def test_tuple_equals_list_discover():
    data = list("ABCABCABC")
    assert Nexora().discover(tuple(data))["count"] == Nexora().discover(data)["count"]


def test_range_and_generator_elementwise():
    assert [r["value"] for r in normalize_observations(range(3))] == [0.0, 1.0, 2.0]
    nx1, nx2 = Nexora(), Nexora()
    assert nx1.discover(range(5))["count"] == nx2.discover(iter([0, 1, 2, 3, 4]))["count"]
    assert nx1.find_anomalies(tuple([10.0] * 30 + [25.0]))["count"] == 1
    assert nx2.quality(iter([1, 2, 3]))["n"] == 3


def test_predict_update_accept_iterables():
    nx = Nexora()
    assert nx.predict(tuple("ABABAB"))["status"] in ("FOUND", "LOW_CONFIDENCE", "NONE")
    out = nx.update(iter([1.0, 2.0, 3.0]))
    assert out["processed"] == 3 and out["total"] == 3
    assert nx.update((4.0, 5.0))["total"] == 5


def test_bare_scalars_rejected_with_hint():
    with pytest.raises(TypeError, match="wrap.*\\[value\\]"):
        normalize_observations("ABC")
    with pytest.raises(TypeError, match="wrap a single row"):
        normalize_observations({"value": 1})
    with pytest.raises(TypeError, match="wrap.*\\[value\\]"):
        parse(5)
    nx = Nexora()
    with pytest.raises(TypeError, match="wrap.*\\[value\\]"):
        nx.quality("abc")
    # ... but the engine still treats one bare scalar as one observation
    assert nx.discover(5)["evidence"] is not None


def test_parse_single_dict_and_generator():
    rows = parse({"value": 7, "label": "G"})
    assert len(rows) == 1 and rows[0]["value"] == 7
    assert [r["value"] for r in parse(iter(["A", "B"]))] == ["A", "B"]
