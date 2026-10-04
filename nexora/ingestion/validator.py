"""Dataset validation for Nexora."""

import math
from typing import Any, Dict, List


def _is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in (
            "", "none", "null", "nan", "na", "n/a"):
        return True
    return False


def validate(rows: Any, max_missing_fraction: float = 0.3) -> Dict[str, Any]:
    """Validate a dataset before detection runs.

    What it computes: dict {"ok": bool, "warnings": list[str],
    "n": int (row count), "missing": int (rows with value None/NaN)}.
    ok is False when empty or when missing/n exceeds
    max_missing_fraction (0..1, default 0.3). A constant series
    (all present values identical) adds a warning but keeps ok True.
    Accepts normalized row dicts or raw scalars; None items count as
    missing, never crash. No 0..1 quality score is produced here.
    """
    try:
        mm = float(max_missing_fraction)
    except (TypeError, ValueError):
        raise ValueError("max_missing_fraction must be a number in 0..1")
    if not 0.0 <= mm <= 1.0:
        raise ValueError("max_missing_fraction must be within 0..1")
    if rows is None:
        return {"ok": False, "warnings": ["empty dataset: no observations"],
                "n": 0, "missing": 0}
    if not isinstance(rows, (list, tuple)):
        raise TypeError("validate() expects a list of rows")
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"ok": False, "warnings": ["empty dataset: no observations"],
                "n": 0, "missing": 0}
    warnings: List[str] = []
    missing = 0
    present: List[Any] = []
    for r in rows:
        if r is None:
            missing += 1
            continue
        if isinstance(r, dict) and "value" in r:
            v = r["value"]
        else:
            v = r
        if _is_missing_value(v):
            missing += 1
        else:
            present.append(v)
    frac = missing / n if n else 0.0
    if missing:
        warnings.append(f"{missing}/{n} values missing ({frac:.1%}).")
    ok = True
    if frac > mm:
        ok = False
        warnings.append(
            f"missing fraction {frac:.1%} exceeds limit {mm:.1%}.")
    if len(present) >= 2:
        first = present[0]
        try:
            if all(v == first for v in present[1:]):
                warnings.append(
                    f"constant series: all {len(present)} present values "
                    "identical; detectors may find nothing.")
        except Exception:
            pass
    return {"ok": ok, "warnings": warnings, "n": n, "missing": missing}
