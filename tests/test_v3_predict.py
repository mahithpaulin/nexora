"""v3 iteration 2: predict(current) conditions context backoff on current.

Data: X A B X A. After the tail (X, A) comes B; after (A, B) comes X.
An explicit current=B must therefore predict X (not the tail's B),
while restating the last label (A) changes nothing. Unknown and
unhashable currents back off honestly instead of answering from the tail.
"""
from nexora.api.engine import Nexora

DATA = ["X", "A", "B", "X", "A"]


def test_explicit_current_conditions_context():
    nx = Nexora()
    base = nx.predict(DATA)
    assert base["evidence"]["context_from"] == "tail"
    assert base["context"][0]["next"] == "B"

    given_b = nx.predict(DATA, current="B")
    assert given_b["current"] == "B"
    assert given_b["evidence"]["context_from"] == "explicit current"
    assert given_b["evidence"]["context_query"][-1] == "B"
    assert given_b["context"][0]["next"] == "X"

    restated = nx.predict(DATA, current="A")
    assert restated["evidence"]["context_from"] == "tail"
    assert [c["next"] for c in restated["context"]] == [c["next"] for c in base["context"]]


def test_unknown_current_backs_off_honestly():
    nx = Nexora()
    zzz = nx.predict(["A", "B", "C"], current="ZZZ")
    assert zzz["evidence"]["context_query"][-1] == "ZZZ"
    assert zzz["status"] == "LOW_CONFIDENCE"


def test_unhashable_current_keeps_tail():
    nx = Nexora()
    base = nx.predict(DATA)
    odd = nx.predict(DATA, current={"never": "hashable"})
    assert odd["evidence"]["context_from"] == "tail"
    assert [c["next"] for c in odd["context"]] == [c["next"] for c in base["context"]]
