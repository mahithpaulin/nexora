"""Isolated batch processing for Nexora (stdlib only).

Isolation: :func:`batch_process` builds a fresh ``Nexora`` per dataset, so
no repository/memory/trail state is shared between runs. Config is
copied per run for the same reason.

Lazy import: ``from nexora.api.engine import Nexora`` happens inside
function bodies (never at module top). This avoids circular imports when
``nexora.api`` imports this module during engine initialisation.
Callers get a clear ``RuntimeError`` if the engine is unavailable.
"""

import collections
import hashlib
import json

__all__ = ["batch_process", "summarize_batch", "compare_signatures"]


def _conf_of(p):
    """Numeric confidence of a pattern dict/object, else None."""
    try:
        c = p.get("confidence", None) if isinstance(p, dict) else getattr(p, "confidence", None)
    except Exception:
        return None
    if isinstance(c, bool):
        return None
    return float(c) if isinstance(c, (int, float)) else None


def _type_of(p):
    """Pattern type string, 'unknown' fallback (never raises)."""
    try:
        t = p.get("type", "unknown") if isinstance(p, dict) else getattr(p, "type", "unknown")
        return str(t) if t is not None else "unknown"
    except Exception:
        return "unknown"


def _id_of(p, fallback):
    """Pattern id string, fallback when missing."""
    try:
        pid = p.get("id", None) if isinstance(p, dict) else getattr(p, "id", None)
        return str(pid) if pid not in (None, "") else fallback
    except Exception:
        return fallback


def _sig_of(p):
    """md5 fingerprint of json({type,features,sequence}).

    NOTE: non-cryptographic use — only a change-detection fingerprint
    for comparing pattern sets, not for security/integrity.
    """
    try:
        if isinstance(p, dict):
            payload = {"type": p.get("type"), "features": p.get("features", {}),
                       "sequence": p.get("sequence", [])}
        else:
            payload = {"type": getattr(p, "type", None),
                       "features": getattr(p, "features", {}),
                       "sequence": getattr(p, "sequence", [])}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        raw = str(p)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def batch_process(datasets, config=None):
    """Run ``detect()`` once per dataset on a fresh isolated engine.

    One bad dataset never kills the batch: each run is wrapped in
    try/except and records ``{"index", "result", "error"}`` with
    ``error`` None on success or ``"ExcType: msg"`` on failure.
    """
    try:
        # Lazy to avoid circulars when nexora.api imports this module.
        from nexora.api.engine import Nexora
    except ImportError as exc:
        raise RuntimeError("nexora.api.engine.Nexora unavailable: %s" % exc) from exc
    if datasets is None:
        return []
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
    outcomes = []
    for i, ds in enumerate(datasets):
        try:
            cfg = dict(config) if isinstance(config, dict) else config
            result = Nexora(cfg).detect(ds)
            outcomes.append({"index": i, "result": result, "error": None})
        except Exception as exc:
            outcomes.append({"index": i, "result": None,
                             "error": "%s: %s" % (type(exc).__name__, exc)})
    return outcomes


def summarize_batch(outcomes):
    """Summarise batch outcomes. Never raises, even on malformed input."""
    try:
        items = list(outcomes) if isinstance(outcomes, (list, tuple)) else []
        n_runs = len(items)
        n_ok, total = 0, 0
        confs, types = [], collections.Counter()
        for o in items:
            try:
                if not isinstance(o, dict):
                    continue
                if o.get("error") is not None:
                    continue
                res = o.get("result")
                if not isinstance(res, dict):
                    continue
                n_ok += 1
                pats = res.get("patterns", [])
                if not isinstance(pats, (list, tuple)):
                    continue
                total += len(pats)
                for p in pats:
                    c = _conf_of(p)
                    if c is not None:
                        confs.append(c)
                    types[_type_of(p)] += 1
            except Exception:
                continue
        n_failed = n_runs - n_ok
        avg_patterns = (total / n_ok) if n_ok else 0.0
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        by_type = dict(sorted(types.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        reason = ("Batch summary: %d/%d ok, %d failed, %d patterns "
                  "(avg %.2f/run, confidence %.3f)." % (n_ok, n_runs, n_failed, total, avg_patterns, avg_conf))
        return {"n_runs": n_runs, "n_ok": n_ok, "n_failed": n_failed,
                "total_patterns": total, "avg_patterns": avg_patterns,
                "avg_confidence": avg_conf, "types": by_type, "reason": reason}
    except Exception as exc:
        return {"n_runs": 0, "n_ok": 0, "n_failed": 0, "total_patterns": 0,
                "avg_patterns": 0.0, "avg_confidence": 0.0, "types": {},
                "reason": "summarize failed: %s: %s" % (type(exc).__name__, exc)}


def compare_signatures(patterns_a, patterns_b):
    """Diff two pattern lists by content signature; report ids.

    Added = b-sigs not in a; removed = a-sigs not in b; common = b ids
    whose sig exists in a. Id lists are sorted for determinism.
    """
    try:
        list_a = list(patterns_a) if isinstance(patterns_a, (list, tuple)) else []
        list_b = list(patterns_b) if isinstance(patterns_b, (list, tuple)) else []
    except Exception:
        list_a, list_b = [], []
    sigs_a, sigs_b = {}, {}
    for p in list_a:
        try:
            s = _sig_of(p)
            sigs_a.setdefault(s, []).append(_id_of(p, "sig:" + s[:8]))
        except Exception:
            continue
    for p in list_b:
        try:
            s = _sig_of(p)
            sigs_b.setdefault(s, []).append(_id_of(p, "sig:" + s[:8]))
        except Exception:
            continue
    set_a, set_b = set(sigs_a), set(sigs_b)
    added = sorted(i for s in (set_b - set_a) for i in sigs_b[s])
    removed = sorted(i for s in (set_a - set_b) for i in sigs_a[s])
    common = sorted(i for s in (set_a & set_b) for i in sigs_b[s])
    reason = "Compared %d vs %d pattern(s): +%d -%d common %d." % (
        len(list_a), len(list_b), len(added), len(removed), len(common))
    return {"added": added, "removed": removed, "common": common, "reason": reason}
