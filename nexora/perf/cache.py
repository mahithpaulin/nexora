"""LRU caching and timing helpers for Nexora (single-threaded, stdlib only).

Complexities: LRUCache get/set O(1) time, O(maxsize) space; memoize adds
O(1) key-building overhead; cached_dtw costs one DTW computation per
cache miss plus O(1) cache ops; timed adds O(1) overhead around fn.
Score ranges: cached_dtw similarity in [0, 1], distance >= 0.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from functools import wraps

__all__ = ["LRUCache", "memoize", "cached_dtw", "timed"]

_SENTINEL = object()


class LRUCache:
    """Fixed-capacity LRU cache backed by OrderedDict.

    Thread-unsafe by design: the engine is single-threaded, so no locks
    are taken; do not share instances across threads without external
    locking. get() refreshes recency; set() inserts/refreshes and evicts
    the least-recently-used entry when over capacity.

    Time O(1) get/set; space O(maxsize).
    """

    def __init__(self, maxsize: int = 128) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize <= 0:
            raise ValueError("maxsize must be a positive int")
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        """Return cached value (refreshing recency) or ``default``. O(1)."""
        try:
            value = self._data.pop(key)
        except KeyError:
            self._misses += 1
            return default
        self._data[key] = value
        self._hits += 1
        return value

    def set(self, key, value) -> None:
        """Insert or overwrite ``key``; evict LRU entry if full. O(1)."""
        if key in self._data:
            del self._data[key]
        self._data[key] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        """Return {"hits","misses","size","maxsize"}. O(1)."""
        return {"hits": self._hits, "misses": self._misses,
                "size": len(self._data), "maxsize": self._maxsize}

    def clear(self) -> None:
        """Empty the cache and reset hit/miss counters. O(1)."""
        self._data.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key) -> bool:
        return key in self._data


def memoize(maxsize: int = 128):
    """Cache a pure function by hashable args/kwargs (LRU, ``maxsize``).

    Key is (args, tuple(sorted(kwargs.items()))); unhashable calls
    bypass the cache. Exposes .cache_stats() and .cache_clear().
    Time O(1) cache overhead per call; space O(maxsize).
    """
    if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize <= 0:
        raise ValueError("maxsize must be a positive int")

    def decorator(fn):
        cache = LRUCache(maxsize)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                key = (args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                return fn(*args, **kwargs)
            result = cache.get(key, _SENTINEL)
            if result is not _SENTINEL:
                return result
            result = fn(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache_stats = cache.stats
        wrapper.cache_clear = cache.clear
        wrapper.cache = cache
        return wrapper

    return decorator


_DTW_CACHE = LRUCache(maxsize=128)


def cached_dtw(a, b, window=None, scale: float = 1.0):
    """Tuple-keyed LRU-cached DTW returning (dist, similarity), no path.

    Why no path: alignment paths are O(n*m) lists that are large and
    rarely reused, so caching them wastes memory; use matching.dtw for
    paths and cache only the compact (dist, similarity) pair here.
    dist >= 0; similarity in [0, 1].

    The DTW implementation is lazily imported inside this function to
    avoid import cycles between nexora.perf and nexora.matching.
    Invalid numeric entries raise ValueError via the underlying
    function. Time: O(1) on cache hit, else cost of dtw_distance;
    space O(maxsize) entries.
    """
    from nexora.matching.dtw import dtw_distance  # lazy: avoid import cycle

    if not isinstance(a, (list, tuple)):
        raise TypeError("a must be a list or tuple of numbers")
    if not isinstance(b, (list, tuple)):
        raise TypeError("b must be a list or tuple of numbers")
    ta, tb = tuple(a), tuple(b)
    key = (ta, tb, window, scale)
    try:
        hash(key)
    except TypeError:
        dist, _path, sim = dtw_distance(ta, tb, window=window, scale=scale)
        return dist, sim
    hit = _DTW_CACHE.get(key, _SENTINEL)
    if hit is not _SENTINEL:
        return hit
    dist, _path, sim = dtw_distance(ta, tb, window=window, scale=scale)
    _DTW_CACHE.set(key, (dist, sim))
    return dist, sim


cached_dtw.cache_stats = _DTW_CACHE.stats
cached_dtw.cache_clear = _DTW_CACHE.clear


def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs); return (result, seconds_float). O(fn)+O(1)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
