"""Data-quality report for raw input lists. Stdlib only (math, statistics).

Formula: quality = max(0.0, 1.0 - missing_fraction - (0.3 if constant else 0.0)
- min(0.3, outlier_fraction)), each fraction in [0, 1], quality in [0, 1].
Self-contained: no nexora imports. Deterministic.
"""
import math
import statistics
VALUE_KEYS = ("value", "v", "x", "val", "y")
def _extract(item):
    """Return dict value-like entry (first of VALUE_KEYS) else item itself."""
    if isinstance(item, dict):
        for k in VALUE_KEYS:
            if k in item:
                return item[k]
        return item
    return item
def _is_missing(val):
    """True for None, float NaN, or empty/whitespace-only str."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False
def _is_numeric(val):
    """True for int/float (excl. bool, excl. NaN); inf counts as numeric."""
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    if isinstance(val, float):
        return not math.isnan(val)
    return False
def _outlier_fraction(nums):
    """Fraction of nums with |z| > 3 vs population mean/pstdev; 0.0 if <2, stdev==0, non-finite, or error."""
    if len(nums) == 0:
        return 0.0
    try:
        fs = [float(v) for v in nums]
        if any(not math.isfinite(v) for v in fs):
            return 0.0
        if len(fs) < 2:
            return 0.0
        m = statistics.mean(fs)
        sd = statistics.pstdev(fs)
        if sd == 0 or not math.isfinite(sd) or not math.isfinite(m):
            return 0.0
        c = sum(1 for v in fs if abs((v - m) / sd) > 3)
        return c / len(fs)
    except Exception:
        return 0.0
def quality_report(data):
    """Assess raw list quality; never raises on ordinary data.

    Accepts list/tuple/None of numbers/strings/None/NaN/dicts (dicts use
    first present key among value/v/x/val/y, else the dict itself); None
    is treated as []. Non-list/tuple/None raises TypeError.
    Returns {n:int, missing:int, missing_fraction:[0,1],
    numeric_fraction:[0,1] of non-missing that are numeric (0.0 if none
    present), constant:bool (all present equal; False if none present),
    outlier_fraction:[0,1] (|z|>3 vs population stats, 0.0 when stdev==0),
    quality:[0,1] via module formula, warnings:[str], reason:str}.
    Empty input gives quality 0.0 with an 'empty data' warning.
    """
    if data is None:
        data = []
    elif isinstance(data, tuple):
        data = list(data)
    elif isinstance(data, list):
        pass
    else:
        raise TypeError("data must be a list, tuple, or None")
    n = len(data)
    if n == 0:
        return {"n": 0, "missing": 0, "missing_fraction": 0.0, "numeric_fraction": 0.0, "constant": False, "outlier_fraction": 0.0, "quality": 0.0, "warnings": ["empty data"], "reason": "empty data"}
    extracted = [_extract(it) for it in data]
    missing = sum(1 for v in extracted if _is_missing(v))
    present = [v for v in extracted if not _is_missing(v)]
    missing_fraction = missing / n if n else 0.0
    numerics = [v for v in present if _is_numeric(v)]
    numeric_fraction = (len(numerics) / len(present)) if present else 0.0
    try:
        if len(present) == 0:
            constant = False
        else:
            first = present[0]
            constant = all(v == first for v in present)
    except Exception:
        constant = False
    outlier_fraction = _outlier_fraction(numerics)
    quality = max(0.0, 1.0 - missing_fraction - (0.3 if constant else 0.0) - min(0.3, outlier_fraction))
    quality = max(0.0, min(1.0, quality))
    warnings = []
    if missing == n:
        warnings.append("all values missing")
    elif missing > 0:
        warnings.append("%d/%d missing values" % (missing, n))
    if constant and len(present) > 0:
        warnings.append("constant data")
    if present:
        if numeric_fraction == 0.0:
            warnings.append("no numeric values")
        elif numeric_fraction < 1.0:
            warnings.append("non-numeric values present")
    if outlier_fraction > 0.0:
        warnings.append("outlier fraction %.3f" % outlier_fraction)
    if n == 0:
        reason = "empty data"
    elif missing == n:
        reason = "all values missing"
    elif constant:
        reason = "constant data"
    elif missing_fraction > 0.5:
        reason = "high missing fraction"
    elif numeric_fraction == 0.0:
        reason = "no numeric values"
    elif numeric_fraction < 0.5:
        reason = "low numeric fraction"
    elif outlier_fraction > 0.1:
        reason = "high outlier fraction"
    elif missing_fraction > 0.0:
        reason = "minor missing data"
    elif outlier_fraction > 0.0:
        reason = "minor outliers"
    else:
        reason = "ok"
    return {"n": n, "missing": missing, "missing_fraction": missing_fraction, "numeric_fraction": numeric_fraction, "constant": bool(constant), "outlier_fraction": outlier_fraction, "quality": quality, "warnings": warnings, "reason": reason}
