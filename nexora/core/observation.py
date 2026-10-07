"""Observation normalization for Nexora.

Converts heterogeneous inputs (scalars, category strings, dict rows,
None/missing) into uniform rows with keys
{"index", "value", "label", "timestamp", "raw"}.
Missing values are preserved as None and never raise.
"""

import math
from typing import Any, Dict, List

_MISSING_TOKENS = {"", "none", "null", "nan", "na", "n/a", "?", "-"}

_VALUE_KEYS = ("value", "v", "x", "val", "y")
_LABEL_KEYS = ("label", "class", "category")
_TS_KEYS = ("timestamp", "t", "time", "ts")


def _is_missing_number(x: Any) -> bool:
    return isinstance(x, float) and (math.isnan(x))


def _coerce_timestamp(ts: Any) -> Any:
    """Return int/float timestamp as-is, parse numeric strings, else None."""
    if ts is None or isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float)):
        if _is_missing_number(ts):
            return None
        return ts
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            f = float(s)
            return None if math.isnan(f) else f
        except ValueError:
            return None
    return None


def _pick(d: Dict[str, Any], keys: tuple) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_observations(data: Any) -> List[Dict[str, Any]]:
    """Normalize a sequence into uniform observation rows.

    What it computes: one dict per input element with keys:
      index (int, position in input), value (float|str|None),
      label (str|None), timestamp (int|float|None), raw (original item).
    Accepts any iterable of observations: list, tuple, range, generator,
    or other sequences/iterables (all iterated element-wise). None input
    yields []. A single scalar (str/int/float/dict) is NOT iterated:
    wrap it as [item] — a TypeError says so. Never crashes on
    None/missing items; they become value=None rows.
    """
    if data is None:
        return []
    if isinstance(data, (str, bytes)):
        raise TypeError(
            "normalize_observations() expects an iterable of observations, "
            "got a bare string; wrap a single value as [value] "
            "(plain strings are also parsed via nexora.ingestion.parser.parse)")
    if isinstance(data, dict):
        raise TypeError(
            "normalize_observations() expects an iterable of observations, "
            "got a bare dict; wrap a single row as [row]")
    if not isinstance(data, (list, tuple)):
        try:
            data = list(data)
        except TypeError:
            raise TypeError(
                "normalize_observations() expects an iterable of observations "
                "(list, tuple, range, generator, ...); wrap a single value "
                "as [value]")
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if item is None:
            rows.append({"index": i, "value": None, "label": None,
                         "timestamp": None, "raw": None})
        elif isinstance(item, bool):
            rows.append({"index": i, "value": float(item),
                         "label": str(item), "timestamp": None, "raw": item})
        elif isinstance(item, (int, float)):
            v = None if _is_missing_number(float(item)) else float(item)
            rows.append({"index": i, "value": v, "label": None,
                         "timestamp": None, "raw": item})
        elif isinstance(item, str):
            s = item.strip()
            if s.lower() in _MISSING_TOKENS:
                rows.append({"index": i, "value": None, "label": None,
                             "timestamp": None, "raw": item})
            else:
                rows.append({"index": i, "value": s, "label": s,
                             "timestamp": None, "raw": item})
        elif isinstance(item, dict):
            v = _pick(item, _VALUE_KEYS)
            if isinstance(v, float) and _is_missing_number(v):
                v = None
            if isinstance(v, str) and v.strip().lower() in _MISSING_TOKENS:
                v = None
            lab = _pick(item, _LABEL_KEYS)
            lab = None if lab is None else str(lab)
            ts = _coerce_timestamp(_pick(item, _TS_KEYS))
            rows.append({"index": i, "value": v, "label": lab,
                         "timestamp": ts, "raw": item})
        else:
            rows.append({"index": i, "value": None, "label": None,
                         "timestamp": None, "raw": item})
    return rows


def validate_observations(observations: Any) -> List[Dict[str, Any]]:
    """Check normalized rows are a non-empty well-formed list.

    What it computes: returns the input list unchanged when valid.
    Raises ValueError on empty input and TypeError/ValueError on
    wrong types or rows missing required keys. No scores involved.
    """
    if observations is None:
        raise ValueError("no observations: input is None")
    if not isinstance(observations, (list, tuple)):
        raise TypeError("observations must be a list of dicts")
    if len(observations) == 0:
        raise ValueError("no observations: empty list")
    required = {"index", "value", "label", "timestamp", "raw"}
    for i, r in enumerate(observations):
        if not isinstance(r, dict):
            raise ValueError(f"row {i} is not a dict")
        if not required.issubset(r.keys()):
            raise ValueError(f"row {i} missing keys {required - set(r.keys())}")
    return list(observations)
