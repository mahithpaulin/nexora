"""Input parsing for Nexora: list / CSV-string / JSON-string -> rows."""

import json
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


def parse(data: Any) -> List[Dict[str, Any]]:
    """Parse input into normalized observation rows.

    What it computes: list of {"index","value","label","timestamp","raw"}
    rows via observation.normalize_observations. Accepts a list (passed
    through), a JSON string encoding a list/dict/scalar, or a CSV string
    (header row optional; empty cells -> None/missing). Returns [] for
    None/empty input; never crashes on missing cells. No scores involved.
    """
    if data is None:
        return []
    if isinstance(data, (list, tuple)):
        return normalize_observations(list(data))
    if not isinstance(data, str):
        raise TypeError("parse() expects a list, CSV string, or JSON string")
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
