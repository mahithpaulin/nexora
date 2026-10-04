"""Pattern repository with JSON file persistence.

Persistence is a plain JSON file (no vector DB / sqlite required).
Dedupe signature uses hashlib.sha256 truncated to 12 hex chars. NOTE:
sha256 is used here only as a non-cryptographic dedupe/fingerprint key,
NOT for any security purpose.

v2 notes:
- get()/all() return deep copies: mutating a returned pattern never
  mutates the store.
- update() silently ignores identity keys {"id", "type", "frequency",
  "occurrences", "sequence", "features"}: a patch can never change a
  pattern's identity. Returns True if pid found else False.
"""
import copy
import hashlib
import json
import os


def _get(p, key, default=None):
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


def _set(p, key, value):
    if isinstance(p, dict):
        p[key] = value
    else:
        try:
            setattr(p, key, value)
        except Exception:
            pass
    return p


def _to_dict(pattern):
    if isinstance(pattern, dict):
        return dict(pattern)
    d = {}
    try:
        d.update(vars(pattern))
    except TypeError:
        pass
    for k in ("id", "type", "features", "sequence", "relationships",
              "frequency", "first_seen", "last_seen", "occurrences",
              "confidence", "similarity", "novelty", "context",
              "metadata", "state"):
        v = _get(pattern, k, None)
        if v is not None and k not in d:
            d[k] = v
    return d


def signature_of(pattern):
    """Dedupe signature: sha256(json({type, features, sequence}))[:12].

    Non-crypto use: fingerprint only, truncated for short keys.
    """
    t = _get(pattern, "type", "unknown")
    f = _get(pattern, "features", {}) or {}
    s = _get(pattern, "sequence", []) or []
    payload = json.dumps({"type": t, "features": f, "sequence": s},
                         sort_keys=True, default=str)
    # Non-crypto use: fingerprint only, truncated for short keys.
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class PatternRepository:
    """In-memory pattern store with JSON persistence and merge-on-duplicate."""

    def __init__(self):
        self._patterns = {}
        self._sig_to_id = {}
        self._counter = 1

    def _defaults(self, d):
        d.setdefault("type", "unknown")
        d.setdefault("features", {})
        d.setdefault("sequence", [])
        d.setdefault("relationships", {})
        d.setdefault("frequency", 1)
        d.setdefault("first_seen", None)
        d.setdefault("last_seen", None)
        d.setdefault("occurrences", [])
        d.setdefault("confidence", 0.5)
        d.setdefault("similarity", 0.0)
        d.setdefault("novelty", 0.0)
        d.setdefault("context", {})
        d.setdefault("metadata", {})
        d.setdefault("state", "NEW")
        return d

    def add(self, pattern):
        """Add pattern; on signature duplicate merge and return existing id."""
        d = self._defaults(_to_dict(pattern))
        sig = signature_of(d)
        if sig in self._sig_to_id:
            pid = self._sig_to_id[sig]
            cur = self._patterns[pid]
            try:
                add_f = int(d.get("frequency", 1) or 1)
            except (TypeError, ValueError):
                add_f = 1
            cur["frequency"] = (cur.get("frequency", 0) or 0) + add_f
            occ = d.get("occurrences") or []
            if occ:
                cur.setdefault("occurrences", []).extend(list(occ))
            try:
                _occ = cur.get("occurrences") or []
                _feats = cur.get("features")
                if isinstance(_feats, dict) and _occ and "count" in _feats:
                    _feats["count"] = len(_occ)
            except Exception:
                pass
            if d.get("last_seen") is not None:
                try:
                    if cur.get("last_seen") is None or d["last_seen"] > cur["last_seen"]:
                        cur["last_seen"] = d["last_seen"]
                except TypeError:
                    cur["last_seen"] = d["last_seen"]
            if cur.get("first_seen") is None and d.get("first_seen") is not None:
                cur["first_seen"] = d["first_seen"]
            try:
                if float(d.get("confidence", 0)) > float(cur.get("confidence", 0)):
                    cur["confidence"] = d["confidence"]
            except (TypeError, ValueError):
                pass
            return pid
        pid = "P-%03d" % self._counter
        self._counter += 1
        d["id"] = pid
        self._patterns[pid] = d
        self._sig_to_id[sig] = pid
        return pid

    def get(self, pid):
        """Deep copy of the stored pattern (None if missing).

        Mutating the result never mutates the store (deepcopy, with a
        dict/list-copy fallback for exotic values).
        """
        p = self._patterns.get(pid)
        if p is None:
            return None
        if isinstance(p, dict):
            try:
                return copy.deepcopy(p)
            except Exception:
                fb = dict(p)
                for k, v in fb.items():
                    if isinstance(v, list):
                        fb[k] = list(v)
                    elif isinstance(v, dict):
                        fb[k] = dict(v)
                return fb
        try:
            return copy.deepcopy(p)
        except Exception:
            return p

    def update(self, pid, patch):
        """Merge patch dict into a stored pattern; returns True if found.

        Identity keys {"id", "type", "frequency", "occurrences",
        "sequence", "features"} are silently ignored: a patch can update
        state, relationships, confidence, scores — never identity.
        Used by the engine to persist lifecycle states, evolution flags,
        and relationship maps without changing the pattern's identity
        (signature, frequency, occurrences untouched).
        """
        cur = self._patterns.get(pid)
        if cur is None or not isinstance(patch, dict):
            return False
        for k, v in patch.items():
            if k in ("id", "type", "frequency", "occurrences", "sequence", "features"):
                continue
            cur[k] = v
        return True

    def all(self):
        """Deep copies of all stored patterns (mutation-safe; see get())."""
        out = []
        for v in self._patterns.values():
            try:
                out.append(copy.deepcopy(v))
            except Exception:
                out.append(dict(v) if isinstance(v, dict) else v)
        return out

    def find_by_type(self, t):
        return [p for p in self.all() if p.get("type") == t]

    def size(self):
        return len(self._patterns)

    def import_patterns(self, patterns):
        """Insert pattern dicts preserving ids; counter follows max id.

        Skips entries without an id. Rebuilds the signature map.
        Returns the number inserted.
        """
        n = 0
        for p in patterns or []:
            if not isinstance(p, dict) or p.get("id") is None:
                continue
            pid = str(p["id"])
            try:
                self._patterns[pid] = copy.deepcopy(p)
            except Exception:
                self._patterns[pid] = dict(p)
            try:
                self._sig_to_id[signature_of(p)] = pid
            except Exception:
                pass
            n += 1
        mx = 0
        for pid in self._patterns:
            try:
                mx = max(mx, int(str(pid).split("-")[1]))
            except (ValueError, IndexError):
                pass
        self._counter = max(self._counter, mx + 1)
        return n

    def prune(self, max_total=1000, drop_states=("RETIRED",)):
        """Drop oldest-first patterns in drop_states while size > max_total.

        Ordering is deterministic: unknown last_seen sorts first, then by
        id string. Returns the removed ids. No-op when size <= max_total.
        """
        if self.size() <= max_total:
            return []
        drop = set(drop_states or ())
        cand = [p for p in self._patterns.values() if p.get("state") in drop]
        cand.sort(key=lambda p: (0 if p.get("last_seen") is None else 1,
                                 str(p.get("last_seen")),
                                 str(p.get("id"))))
        removed = []
        while self.size() > max_total and cand:
            pid = cand.pop(0)["id"]
            if pid in self._patterns:
                del self._patterns[pid]
            for sig, mapped in list(self._sig_to_id.items()):
                if mapped == pid:
                    del self._sig_to_id[sig]
            removed.append(pid)
        return removed

    def save(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"counter": self._counter,
                       "patterns": list(self._patterns.values())},
                      f, indent=2, default=str)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("patterns", []) if isinstance(data, dict) else data
        self._patterns = {}
        self._sig_to_id = {}
        for p in items:
            if not isinstance(p, dict) or p.get("id") is None:
                continue
            self._patterns[p["id"]] = dict(p)
            self._sig_to_id[signature_of(p)] = p["id"]
        counter = data.get("counter") if isinstance(data, dict) else None
        if counter is None:
            mx = 0
            for pid in self._patterns:
                try:
                    mx = max(mx, int(str(pid).split("-")[1]))
                except (ValueError, IndexError):
                    pass
            counter = mx + 1
        self._counter = int(counter)
        return self.size()
