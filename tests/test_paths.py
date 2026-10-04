"""File-path ingestion tests: CSV/JSON/gz files + PathLike + engine wiring."""
import gzip
import os

import pytest

from nexora.ingestion.parser import parse
from nexora.api.engine import Nexora


def test_parse_csv_file(tmp_path):
    f = tmp_path / "sensor.csv"
    f.write_text("value,label\n1,A\n2,B\n3,C\n")
    rows = parse(str(f))
    assert [r["value"] for r in rows] == [1, 2, 3]
    assert rows[0]["label"] == "A"


def test_parse_json_file_and_pathlike(tmp_path):
    f = tmp_path / "data.json"
    f.write_text('[{"value": 10}, {"value": 20}]')
    assert [r["value"] for r in parse(f)] == [10, 20]  # PathLike
    assert [r["value"] for r in parse(str(f))] == [10, 20]


def test_parse_gz_file(tmp_path):
    f = tmp_path / "data.csv.gz"
    with gzip.open(str(f), "wt", encoding="utf-8") as fh:
        fh.write("value\n4\n5\n")
    assert [r["value"] for r in parse(str(f))] == [4, 5]


def test_parse_missing_pathlike_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse(tmp_path / "nope.csv")


def test_engine_discovers_from_file(tmp_path):
    f = tmp_path / "cycle.csv"
    f.write_text("value\n" + "\n".join(["A", "B", "C"] * 6) + "\n")
    nx = Nexora(config={"min_support": 2, "max_n": 3})
    disc = nx.discover(str(f))
    seqs = [tuple(p["sequence"]) for p in disc["patterns"]]
    assert ("A", "B", "C") in seqs
    anom = nx.find_anomalies(str(f))
    assert anom["count"] >= 0
    rep = nx.report(nx.detect(str(f)))
    assert "Nexora report" in rep


def test_plain_strings_still_observations():
    nx = Nexora(config={"min_support": 5})
    m = nx.match("A")  # no repo patterns -> empty ranking, no crash
    assert m == []
    rows = parse("A")  # inline content, not a file
    assert rows[0]["value"] == "A"
