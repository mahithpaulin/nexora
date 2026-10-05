"""Nexora engine-state persistence (schema v2).

Schema v2 JSON object::

    {
      "version": 2,
      "saved_at": "<UTC ISO-8601>",
      "config": {...},
      "patterns": [...],
      "trails": {"pid": [...]},
      "extra": {...}
    }

MIGRATION (documented): version-1 files (v1.x shape, patterns without
the ``significance`` subdict, fewer config keys) load successfully
and are normalized to version 2 with ``extra={"migrated_from": 1}``
and a reason describing the migration; missing config keys are
filled with current defaults by ``validate_config`` on engine load.
Version-0 files shaped ``{counter, patterns}`` (the v0.1 repo
format, no ``version`` key) migrate the same way with
``config={}``, ``trails={}``. Unknown future versions (> 2) raise
``ValueError``. Missing keys default to empty containers.

Writes are atomic: payload goes to ``path + ".tmp"`` then
``os.replace`` onto ``path``, so a crash never leaves a half-written
state file (POSIX atomic rename; documented best-effort on Windows).
"""

import datetime
import json
import os

SCHEMA_VERSION = 2


def _utc_now_iso():
    """Current UTC time as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pattern_sort_key(p):
    try:
        if isinstance(p, dict):
            return str(p.get("id", ""))
        return str(getattr(p, "id", ""))
    except Exception:
        return ""


def _to_jsonable(p):
    """Best-effort conversion of one pattern to a JSON-able dict."""
    try:
        if isinstance(p, dict):
            return dict(p)
        to_dict = getattr(p, "to_dict", None)
        if callable(to_dict):
            try:
                d = to_dict()
                if isinstance(d, dict):
                    return dict(d)
            except Exception:
                pass
        if hasattr(p, "__dict__"):
            try:
                return dict(vars(p))
            except Exception:
                pass
        return {"repr": str(p)}
    except Exception:
        try:
            return {"repr": str(p)}
        except Exception:
            return {"repr": "?"}


def save_engine_state(path, repository, trails=None, config=None, extra=None):
    """Save engine state to ``path`` as deterministic JSON. Returns ``path``.

    Writes are atomic (tmp file + os.replace). Patterns sorted by id,
    trail keys sorted, payload keys sorted.
    """
    cfg = dict(config) if isinstance(config, dict) else {}
    ext = dict(extra) if isinstance(extra, dict) else {}

    if hasattr(repository, "all") and callable(getattr(repository, "all")):
        try:
            raw = repository.all()
        except Exception:
            raw = []
    else:
        raw = repository

    if raw is None:
        raw_list = []
    elif isinstance(raw, dict):
        raw_list = list(raw.values())
    elif isinstance(raw, (list, tuple)):
        raw_list = list(raw)
    else:
        try:
            raw_list = list(raw)
        except Exception:
            raw_list = []

    patterns = [_to_jsonable(p) for p in raw_list]
    try:
        patterns = sorted(patterns, key=_pattern_sort_key)
    except Exception:
        pass

    if isinstance(trails, dict):
        norm_trails = {}
        try:
            keys = sorted(trails.keys(), key=lambda k: str(k))
        except Exception:
            keys = list(trails.keys())
        for k in keys:
            v = trails[k]
            if isinstance(v, (list, tuple)):
                norm_trails[str(k)] = list(v)
            elif v is None:
                norm_trails[str(k)] = []
            else:
                norm_trails[str(k)] = [v]
    elif trails is None:
        norm_trails = {}
    else:
        try:
            d = dict(trails)
            norm_trails = {}
            for k in sorted(d.keys(), key=lambda x: str(x)):
                v = d[k]
                if isinstance(v, (list, tuple)):
                    norm_trails[str(k)] = list(v)
                elif v is None:
                    norm_trails[str(k)] = []
                else:
                    norm_trails[str(k)] = [v]
        except Exception:
            norm_trails = {}

    payload = {
        "version": SCHEMA_VERSION,
        "saved_at": _utc_now_iso(),
        "config": cfg,
        "patterns": patterns,
        "trails": norm_trails,
        "extra": ext,
    }

    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, str(path))
    return path


def _as_pattern_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        return list(v)
    except Exception:
        return []


def load_engine_state(path):
    """Load engine state file. Returns version/config/patterns/trails/extra/reason.

    Raises ``FileNotFoundError`` if missing, ``ValueError`` on bad JSON,
    missing version (non-migratable), bad version type, or future version.
    """
    if not os.path.exists(str(path)):
        raise FileNotFoundError("state file not found: {}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("invalid state file: top-level JSON must be an object")

    if "version" not in data:
        # MIGRATION: v0.1 repo format {counter, patterns}, no version key.
        if "patterns" in data and "counter" in data:
            pats = _as_pattern_list(data.get("patterns", []))
            return {
                "version": 2,
                "config": {},
                "patterns": pats,
                "trails": {},
                "extra": {"migrated_from": 0},
                "reason": "migrated from version 0 (v0.1 repo format: {counter, patterns})",
            }
        raise ValueError("invalid state file: missing 'version'")

    ver = data.get("version")
    if isinstance(ver, bool) or not isinstance(ver, int):
        raise ValueError("invalid state file: bad version {!r}".format(ver))
    if ver > SCHEMA_VERSION:
        raise ValueError(
            "unsupported state version {} (max supported {})".format(ver, SCHEMA_VERSION)
        )
    if ver == 0:
        # Explicit version-0 file, same v0.1 shape.
        pats = _as_pattern_list(data.get("patterns", []))
        return {
            "version": 2,
            "config": {},
            "patterns": pats,
            "trails": {},
            "extra": {"migrated_from": 0},
            "reason": "migrated from version 0 (v0.1 repo format: {counter, patterns})",
        }

    migrated_from = None
    if ver == 1:
        # MIGRATION: v1 files predate the significance subdict and the
        # v2 config keys; patterns pass through untouched, config gaps
        # are filled by validate_config on engine load.
        migrated_from = 1
    elif ver != 2:
        raise ValueError("unsupported state version {}".format(ver))

    cfg = data.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    pats = data.get("patterns", [])
    if not isinstance(pats, list):
        try:
            pats = list(pats)
        except Exception:
            pats = []
    tr = data.get("trails", {})
    if not isinstance(tr, dict):
        tr = {}
    else:
        try:
            tr = {str(k): (list(v) if isinstance(v, (list, tuple)) else []) for k, v in sorted(tr.items(), key=lambda kv: str(kv[0]))}
        except Exception:
            try:
                tr = {str(k): v for k, v in tr.items()}
            except Exception:
                tr = {}
    ext = data.get("extra", {})
    if not isinstance(ext, dict):
        ext = {}
    if migrated_from is not None:
        ext = dict(ext)
        ext["migrated_from"] = migrated_from
        reason = "migrated from version %d to schema v2" % migrated_from
    else:
        reason = data.get("reason", "ok")
    if not isinstance(reason, str):
        try:
            reason = str(reason)
        except Exception:
            reason = "ok"
    return {
        "version": 2,
        "config": cfg,
        "patterns": pats,
        "trails": tr,
        "extra": ext,
        "reason": reason,
    }
