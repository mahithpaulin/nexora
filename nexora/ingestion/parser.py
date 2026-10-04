"""Input parsing for Nexora: list / CSV-string / JSON-string / file path -> rows."""

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from nexora.core.observation import normalize_observations

_MISSING = {"", "none", "null", "nan", "na", "n/a", "?", "-"}
_KNOWN_HEADERS = {"value", "label", "t", "timestamp", "v", "time", "index"}


def _coerce_cell(cell: Any) -> Any:
    """Coerce one CSV cell: missing tokens -> None, numbers -> int/float."""
    if cell is None:
        return None
    s = str(cell).strip()
    if s.lower() in _MISSING:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _is_header(cells: List[str]) -> bool:
    """Check whether a CSV row looks like a header row."""
    return any(c.strip().lower() in _KNOWN_HEADERS for c in cells)


def _read_text_file(path: str) -> str:
    """Read a text file: utf-8 first, latin-1 fallback; transparent .gz.

    Raises FileNotFoundError when missing (fail-fast, documented).
    """
    if path.endswith(".gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with gzip.open(path, "rt", encoding="latin-1") as f:
                return f.read()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def parse(data: Any) -> List[Dict[str, Any]]:
    """Parse input into normalized observation rows.

    What it computes: list of {"index","value","label","timestamp","raw"}
    rows via observation.normalize_observations. Accepts a list (passed
    through), a JSON string encoding a list/dict/scalar, a CSV string
    (header row optional; empty cells -> None/missing), or a file path
    (str pointing at an existing file, or os.PathLike): the file is read
    (utf-8, latin-1 fallback, transparent .gz) and parsed by extension
    (.json -> JSON, anything else -> content sniff). Returns [] for
    None/empty input; never crashes on missing cells. Raises
    FileNotFoundError for PathLike that does not exist. No scores involved.
    """
    if data is None:
        return []
    if isinstance(data, (list, tuple)):
        return normalize_observations(list(data))
    if isinstance(data, os.PathLike):
        return parse(_read_text_file(os.fspath(data)))
    if isinstance(data, str) and os.path.exists(data) and os.path.isfile(data):
        text = _read_text_file(data)
        if data.endswith(".json") or data.endswith(".json.gz"):
            return parse(text)  # JSON content path
        return parse(text)
    if not isinstance(data, str):
        raise TypeError("parse() expects a list, CSV/JSON string, or file path")
    s = data.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return normalize_observations(obj)
        if isinstance(obj, dict):
            return normalize_observations([obj])
        return normalize_observations([obj])
    except (ValueError, json.JSONDecodeError):
        pass
    lines = [ln for ln in s.splitlines() if ln.strip() != ""]
    if not lines:
        return []
    grid = [[c.strip() for c in ln.split(",")] for ln in lines]
    if _is_header(grid[0]):
        header = [c.strip().lower() for c in grid[0]]
        if len(header) == 1:
            # Single column: header is just a name; rows are scalar values
            # (so strings keep themselves as labels, numbers as values).
            return normalize_observations(
                [_coerce_cell(r[0]) if r else None for r in grid[1:]])
        dicts = []
        for row in grid[1:]:
            d: Dict[str, Any] = {}
            for h, c in zip(header, row):
                d[h] = _coerce_cell(c)
            dicts.append(d)
        return normalize_observations(dicts)
    if len(grid) == 1 and len(grid[0]) > 1:
        return normalize_observations([_coerce_cell(c) for c in grid[0]])
    if all(len(r) == 1 for r in grid):
        return normalize_observations([_coerce_cell(r[0]) for r in grid])
    dicts = []
    for row in grid:
        d = {}
        if len(row) >= 1:
            d["value"] = _coerce_cell(row[0])
        if len(row) >= 2:
            cell = row[1].strip()
            d["label"] = None if cell.lower() in _MISSING or cell == "" else cell
        if len(row) >= 3:
            d["t"] = _coerce_cell(row[2])
        dicts.append(d)
    return normalize_observations(dicts)
