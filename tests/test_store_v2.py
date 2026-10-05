"""Workstream 10: schema-v2 persistence with v1 migration.

A realistic v1 file (version 1, patterns without the significance
subdict, v1-era config keys) must load, normalize to schema v2, and
leave a working engine. v2 roundtrips preserve significance metadata
and config; saves are byte-deterministic apart from saved_at.
"""
import json

from nexora import Nexora
from nexora.memory.store import SCHEMA_VERSION, load_engine_state


def _v1_file(tmp_path):
    """Hand-built v1-shaped state (as v1.0.0 would have written it)."""
    pats = [
        {"id": "P-001", "type": "recurring_value",
         "features": {"value": "A", "count": 4}, "sequence": ["A"],
         "relationships": {}, "frequency": 4,
         "first_seen": 0, "last_seen": 9, "occurrences": [0, 3, 6, 9],
         "confidence": 0.8, "similarity": 1.0, "novelty": 0.0,
         "context": {}, "metadata": {}, "state": "OBSERVED"},
        {"id": "P-002", "type": "frequent_sequence",
         "features": {"sequence": ["A", "B", "C"], "support": 3},
         "sequence": ["A", "B", "C"],
         "relationships": {}, "frequency": 3,
         "first_seen": 0, "last_seen": 6, "occurrences": [0, 3, 6],
         "confidence": 0.7, "similarity": 1.0, "novelty": 0.0,
         "context": {}, "metadata": {}, "state": "OBSERVED"},
    ]
    payload = {"version": 1, "saved_at": "2026-10-04T00:00:00+00:00",
               "config": {"z_threshold": 3.0, "min_support": 3, "max_n": 3},
               "patterns": pats, "trails": {"P-001": [0.8]}, "extra": {}}
    f = str(tmp_path / "v1.json")
    with open(f, "w") as fh:
        json.dump(payload, fh)
    return f


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_v1_file_loads_and_migrates(tmp_path):
    f = _v1_file(tmp_path)
    mig = load_engine_state(f)
    assert mig["version"] == 2
    assert mig["extra"]["migrated_from"] == 1
    assert "migrated from version 1" in mig["reason"]
    assert len(mig["patterns"]) == 2
    nx = Nexora()
    info = nx.load(f)
    assert info["version"] == 2
    assert info["patterns"] == 2
    # Old config keys filled with current defaults (fail-fast intact).
    assert nx.config["min_support"] == 3
    assert nx.config["sig_min_n"] == 30
    assert nx.config["change_z"] == 6.0
    # Engine works on the migrated state.
    assert nx.get_pattern("P-002")["sequence"] == ["A", "B", "C"]
    assert nx.get_history("P-001")["status"] == "FOUND"


def test_v2_roundtrip_preserves_significance(tmp_path):
    nx = Nexora()
    nx.discover(list("ABC") * 12)
    f = str(tmp_path / "v2.json")
    nx.save(f)
    with open(f) as fh:
        payload = json.load(fh)
    assert payload["version"] == 2
    assert any("significance" in p for p in payload["patterns"])
    nx2 = Nexora()
    nx2.load(f)
    p1 = nx.get_pattern("P-001")
    p2 = nx2.get_pattern("P-001")
    assert p1 == p2
    assert p2["significance"]["p_value"] == p1["significance"]["p_value"]


def test_save_bytes_deterministic_apart_from_timestamp(tmp_path):
    nx = Nexora(config={"min_support": 2})
    nx.discover(list("ABCABC"))
    f1, f2 = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    nx.save(f1)
    nx.save(f2)
    d1 = json.load(open(f1))
    d2 = json.load(open(f2))
    d1.pop("saved_at")
    d2.pop("saved_at")
    assert d1 == d2
