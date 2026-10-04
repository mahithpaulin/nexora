"""Pattern lifecycle: NEW -> OBSERVED -> CONFIRMED -> ESTABLISHED (-> EVOLVING)
-> STALE -> RETIRED. All thresholds configurable via config dict."""
STATES = ("NEW", "OBSERVED", "CONFIRMED", "ESTABLISHED", "EVOLVING", "STALE", "RETIRED")

DEFAULT_CONFIG = {
    "confirm_threshold": 3,      # OBSERVED freq >= 3 -> CONFIRMED
    "establish_freq": 10,        # CONFIRMED freq >= 10 -> ESTABLISHED
    "establish_confidence": 0.8,  # ... or confidence >= 0.8 -> ESTABLISHED
    "stale_after": 100,          # idle steps without observation -> STALE
    "retired_after": 200,        # idle steps while STALE -> RETIRED
    "drift_threshold": 0.5,      # drift score >= this -> EVOLVING
}


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


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def advance(pattern, observed=False, config=None):
    """Advance lifecycle state. Returns (new_state, explanation)."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    state = str(_get(pattern, "state", "NEW") or "NEW").upper()
    if state not in STATES:
        state = "NEW"
    try:
        freq = int(_get(pattern, "frequency", 0) or 0)
    except (TypeError, ValueError):
        freq = 0
    conf = _num(_get(pattern, "confidence", 0.0) or 0.0)
    idle = _get(pattern, "idle_steps", None)
    if idle is None:
        idle = _get(pattern, "steps_since_seen", 0) or 0
    try:
        idle = int(idle)
    except (TypeError, ValueError):
        idle = 0
    drift = _get(pattern, "drift", None)
    if drift is None:
        drift = _get(pattern, "drift_score", _get(pattern, "novelty", 0.0) or 0.0)
    drift = _num(drift, 0.0)
    c_thr = cfg["confirm_threshold"]
    e_fr = cfg["establish_freq"]
    e_cf = cfg["establish_confidence"]
    st = cfg["stale_after"]
    rt = cfg["retired_after"]
    d_thr = cfg["drift_threshold"]

    if observed:
        idle = 0
        if state == "RETIRED":
            new, why = "OBSERVED", "re-observed after RETIRED; revived to OBSERVED"
        elif state == "STALE":
            new, why = "OBSERVED", "re-observed after STALE; reactivated to OBSERVED"
        elif state == "NEW":
            new, why = "OBSERVED", "first observation received; NEW -> OBSERVED"
        elif state in ("CONFIRMED", "ESTABLISHED") and drift >= d_thr:
            new, why = "EVOLVING", ("significant feature drift %.2f >= %.2f; %s -> EVOLVING" % (drift, d_thr, state))
        elif state == "EVOLVING":
            if drift >= d_thr:
                new, why = "EVOLVING", ("drift %.2f still >= %.2f; remains EVOLVING" % (drift, d_thr))
            else:
                new, why = "CONFIRMED", ("drift %.2f < %.2f; EVOLVING -> CONFIRMED" % (drift, d_thr))
        elif state == "OBSERVED":
            if freq >= c_thr:
                new, why = "CONFIRMED", ("frequency %d >= %d; OBSERVED -> CONFIRMED" % (freq, c_thr))
            else:
                new, why = "OBSERVED", ("frequency %d < %d; remains OBSERVED" % (freq, c_thr))
        elif state == "CONFIRMED":
            if freq >= e_fr or conf >= e_cf:
                new, why = "ESTABLISHED", ("frequency %d (need %d) or confidence %.2f (need %.2f); CONFIRMED -> ESTABLISHED" % (freq, e_fr, conf, e_cf))
            else:
                new, why = "CONFIRMED", ("frequency %d < %d and confidence %.2f < %.2f; remains CONFIRMED" % (freq, e_fr, conf, e_cf))
        else:  # ESTABLISHED
            new, why = "ESTABLISHED", ("frequency %d, confidence %.2f; remains ESTABLISHED" % (freq, conf))
    else:
        idle = idle + 1
        if state == "RETIRED":
            new, why = "RETIRED", ("already RETIRED; idle %d steps" % idle)
        elif state == "STALE" and idle >= rt:
            new, why = "RETIRED", ("idle %d >= retired_after %d; STALE -> RETIRED" % (idle, rt))
        elif state not in ("STALE", "RETIRED") and idle >= st:
            if state in ("CONFIRMED", "ESTABLISHED") and drift >= d_thr:
                new, why = "EVOLVING", ("drift %.2f >= %.2f without fresh observation; %s -> EVOLVING" % (drift, d_thr, state))
            else:
                new, why = "STALE", ("no observation for %d steps (>= stale_after %d); %s -> STALE" % (idle, st, state))
        elif state in ("CONFIRMED", "ESTABLISHED") and drift >= d_thr:
            new, why = "EVOLVING", ("drift %.2f >= %.2f; %s -> EVOLVING" % (drift, d_thr, state))
        else:
            new, why = state, ("no observation (idle %d); remains %s" % (idle, state))
    _set(pattern, "state", new)
    _set(pattern, "idle_steps", idle)
    expl = "Lifecycle: %s -> %s (%s)." % (state, new, why)
    return new, expl


def update_confidence(pattern, matched):
    """Update confidence in place. Returns (new_confidence, explanation)."""
    c = _num(_get(pattern, "confidence", 0.5) or 0.0)
    if matched:
        new = min(1.0, c + 0.05)
        expl = "Matched observation; confidence %.3f -> %.3f (+0.05, capped at 1.0)." % (c, new)
    else:
        new = max(0.0, c - 0.10)
        expl = "Missed observation; confidence %.3f -> %.3f (-0.10, floored at 0.0)." % (c, new)
    _set(pattern, "confidence", new)
    return new, expl


class Lifecycle:
    """Convenience wrapper with configurable thresholds."""

    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    def advance(self, pattern, observed=False, config=None):
        cfg = dict(self.config)
        if config:
            cfg.update(config)
        return advance(pattern, observed, cfg)

    def update_confidence(self, pattern, matched):
        return update_confidence(pattern, matched)
