"""Nexora efficiency run: wall-clock + peak-memory scaling (stdlib only).

Sizes 1k/5k/20k on two-regime data: discover, find_anomalies, predict,
stream_anomalies, plus cached-vs-uncached DTW. Memory via tracemalloc
peak for discover. Deterministic data; timings indicative, not asserts.

Run: python examples/efficiency.py   (from repo root)
"""
import sys
import time
import tracemalloc

sys.path.insert(0, ".")

from nexora import Nexora
from nexora.matching.dtw import dtw_distance
from nexora.perf.cache import cached_dtw, timed
from nexora.streaming import stream_anomalies


def regime_data(n):
    half = n // 2
    return [5.0 + (i % 4) * 0.1 for i in range(half)] + \
           [50.0 - (i % 4) * 0.1 for i in range(n - half)]


print(f"{'size':>6} | {'discover':>9} | {'anomalies':>9} | {'predict':>8} | {'stream':>8} | {'peak MB':>7}")
for n in (1000, 5000, 20000):
    d = regime_data(n)
    nx = Nexora()
    _, t_disc = timed(nx.discover, d)
    tracemalloc.start()
    nx2 = Nexora()
    nx2.discover(d)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _, t_an = timed(nx.find_anomalies, d)
    _, t_pr = timed(nx.predict, d)
    _, t_st = timed(stream_anomalies, d, 100)
    print(f"{n:>6} | {t_disc:>8.2f}s | {t_an:>8.2f}s | {t_pr:>7.2f}s | {t_st:>7.2f}s | {peak/1e6:>6.1f}")

a, b = [float(i % 7) for i in range(100)], [float((i + 1) % 7) for i in range(100)]
cached_dtw.cache_clear()
((d0, _, _), t1) = timed(dtw_distance, a, b)
(_, _), t2 = timed(cached_dtw, a, b)
(_, _), t3 = timed(cached_dtw, a, b)
print(f"\nDTW 100x100: uncached {t1:.3f}s, miss {t2:.3f}s, hit {t3:.4f}s "
      f"(hit {t2/max(t3,1e-9):.0f}x faster), cache={cached_dtw.cache_stats()}")
print("Done.")
