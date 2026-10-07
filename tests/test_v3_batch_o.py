"""Loop 2 batch O (I76-I80)."""
import json

import pytest

from nexora.api.engine import Nexora


def test_snapshot_restore():
    a = Nexora()
    a.discover(list("ABCABCABC"))
    snap = a.snapshot()["snapshot"]
    json.dumps(snap)
    b = Nexora()
    assert b.restore(snap)["patterns"] == a.top_patterns(1000)["count"]
    assert b.state_signature()["signature"] == a.state_signature()["signature"]
    with pytest.raises(ValueError):
        b.restore({"v": 2})


def test_share_adopt():
    a = Nexora()
    assert a.share("P-999")["status"] == "NONE"
    a.discover(list("ABCABCABC"))
    pid = a.top_patterns(1)["patterns"][0]["id"]
    shared = a.share(pid)
    assert shared["status"] == "FOUND"
    json.dumps(shared["pattern"])
    b = Nexora()
    assert b.adopt(shared["pattern"])["id"] == pid
    assert b.get_pattern(pid) is not None
    with pytest.raises(ValueError):
        b.adopt({"noid": 1})


def test_merge():
    a, b = Nexora(), Nexora()
    a.discover(list("ABCABCABC"))
    b.discover(list("XYZXYZXYZ"))
    m = a.merge(b)
    assert m["status"] == "FOUND" and m["added"] >= 1
    assert a.merge([])["added"] == 0
