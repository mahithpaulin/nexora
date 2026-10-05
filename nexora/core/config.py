"""Configuration defaults and validation for Nexora.

Self-contained: tries ``from nexora.core.errors import InvalidConfigError``
and falls back to a locally defined ``InvalidConfigError`` on ImportError,
so this module works standalone with no other nexora imports. Only stdlib
``math`` is used.
"""
try:
    from nexora.core.errors import InvalidConfigError
except ImportError:  # standalone fallback; keeps module self-contained
    class InvalidConfigError(ValueError):
        """Raised when a config value is invalid."""
        def __init__(self, message="", key=None):
            super().__init__(message)
            self.key = key
import math
DEFAULTS = {
    "z_threshold": 3.0,
    "min_support": 3,
    "max_n": 3,
    "window": 20,
    "weights": {},
    "prune_redundant": True,
    "regimes": True,
    "regime_size": 8,
    "n_clusters": 2,
    "correlation": True,
    "corr_threshold": 0.7,
    "seasonality": True,
    "context_order": 2,
    "multivariate": True,
    "mv_window": 5,
    "mv_threshold": 0.8,
    "evolve_drift": 0.4,
    "max_period": 256,
}
SCHEMA = {
    "z_threshold": ((float, int), 0.1, 10),
    "min_support": (int, 1, 10**6),
    "max_n": (int, 2, 10),
    "window": (int, 1, 10**6),
    "weights": (dict, None, None),
    "prune_redundant": (bool, None, None),
    "regimes": (bool, None, None),
    "regime_size": (int, 2, 10**4),
    "n_clusters": (int, 1, 10**3),
    "correlation": (bool, None, None),
    "corr_threshold": ((float, int), 0, 1),
    "seasonality": (bool, None, None),
    "context_order": (int, 0, 10),
    "multivariate": (bool, None, None),
    "mv_window": (int, 2, 10**4),
    "mv_threshold": ((float, int), 0, 1),
    "evolve_drift": ((float, int), 0, 1),
    "max_period": (int, 2, 10**6),
}
KEY_DOCS = {
    "z_threshold": "Z-score cutoff for flagging anomalies, range [0.1, 10].",
    "min_support": "Minimum occurrences for a pattern to be kept, range [1, 1000000].",
    "max_n": "Maximum n-gram/order examined, range [2, 10].",
    "window": "Sliding window length for streaming stats, range [1, 1000000].",
    "weights": "Optional per-symbol weight map; must be a dict.",
    "prune_redundant": "Drop sub-patterns subsumed by a longer pattern with equal support; must be bool.",
    "regimes": "Enable regime segmentation if True; must be bool.",
    "regime_size": "Minimum points per regime segment, range [2, 10000].",
    "n_clusters": "Number of clusters for regime grouping, range [1, 1000].",
    "correlation": "Enable cross-signal correlation if True; must be bool.",
    "corr_threshold": "Correlation magnitude threshold, range [0, 1].",
    "seasonality": "Enable seasonality detection if True; must be bool.",
    "context_order": "Markov context order, range [0, 10].",
    "multivariate": "Enable multivariate modelling if True; must be bool.",
    "mv_window": "Multivariate window length, range [2, 10000].",
    "mv_threshold": "Multivariate agreement threshold, range [0, 1].",
    "evolve_drift": "Allowed drift per evolve step, range [0, 1].",
    "max_period": "Maximum seasonal period examined, range [2, 1000000].",
}
def validate_config(d=None):
    """Validate a config dict and return a fresh validated dict.

    Ranges: z_threshold [0.1, 10]; min_support [1, 1000000]; max_n [2, 10];
    window [1, 1000000]; weights dict; regimes/correlation/seasonality/
    multivariate/prune_redundant bool (exact type, int rejected); regime_size [2, 10000];
    n_clusters [1, 1000]; corr_threshold/mv_threshold/evolve_drift [0, 1];
    context_order [0, 10]; mv_window [2, 10000]; max_period [2, 1000000].
    Bounds are inclusive.
    Starts from DEFAULTS, overlays known keys of ``d``; unknown keys are
    ignored silently. Int fields accept lossless int-valued floats
    (3.0 -> 3) else raise; float fields accept int (coerced to float) and
    finite float, rejecting bool/NaN/inf; raises InvalidConfigError
    (with ``key`` set) on any violation. ``None`` means defaults.
    """
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise InvalidConfigError("config must be a dict, got %s" % type(d).__name__, key="config")
    config = {}
    for k, v in DEFAULTS.items():
        config[k] = dict(v) if isinstance(v, dict) else v
    for key, raw in d.items():
        if key not in DEFAULTS:
            continue
        exp, lo, hi = SCHEMA[key]
        if exp is bool:
            if type(raw) is not bool:
                raise InvalidConfigError("invalid %r: expected bool, got %s" % (key, type(raw).__name__), key=key)
            config[key] = raw
        elif exp is dict:
            if not isinstance(raw, dict):
                raise InvalidConfigError("invalid %r: expected dict, got %s" % (key, type(raw).__name__), key=key)
            config[key] = dict(raw)
        elif exp is int:
            if type(raw) is bool:
                raise InvalidConfigError("invalid %r: expected int, got bool" % key, key=key)
            if isinstance(raw, int):
                v = raw
            elif isinstance(raw, float):
                if not math.isfinite(raw) or not raw.is_integer():
                    raise InvalidConfigError("invalid %r: expected int, got non-integral float %r" % (key, raw), key=key)
                v = int(raw)
            else:
                raise InvalidConfigError("invalid %r: expected int, got %s" % (key, type(raw).__name__), key=key)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                raise InvalidConfigError("invalid %r: %r out of range [%r, %r]" % (key, v, lo, hi), key=key)
            config[key] = v
        elif exp == (float, int):
            if type(raw) is bool:
                raise InvalidConfigError("invalid %r: expected number, got bool" % key, key=key)
            if isinstance(raw, int):
                v = float(raw)
            elif isinstance(raw, float):
                if not math.isfinite(raw):
                    raise InvalidConfigError("invalid %r: expected finite float, got %r" % (key, raw), key=key)
                v = float(raw)
            else:
                raise InvalidConfigError("invalid %r: expected number, got %s" % (key, type(raw).__name__), key=key)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                raise InvalidConfigError("invalid %r: %r out of range [%r, %r]" % (key, v, lo, hi), key=key)
            config[key] = v
        else:
            raise InvalidConfigError("no schema for %r" % key, key=key)
    return config
