"""Nexora v1.0.0 evaluation: 10 problems with known ground truth.

Each problem asserts a concrete, numerically-margined expectation and
prints PASS/FAIL with the key numbers. Exit 0 iff 10/10 pass.
Deterministic (fixed literals or random.Random(42)).

Run: python examples/eval_10.py   (from repo root; stdlib only)
"""
import math
import random
import sys
import time

sys.path.insert(0, ".")

from nexora import Nexora
from nexora.anomaly.multivariate import detect_multivariate
from nexora.discovery.change_points import change_points
from nexora.discovery.sequences import find_frequent_sequences
from nexora.features.correlation import find_correlation_patterns
from nexora.features.seasonality import estimate_period

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def t(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0

# 1. Repeating cycle ------------------------------------------------
d = list("ABC" * 10)
nx = Nexora(config={"min_support": 2, "max_n": 3})
disc, dt1 = t(nx.discover, d)
abc = [p for p in disc["patterns"] if tuple(p["sequence"]) == ("A", "B", "C")]
pred = nx.predict(d)
ok = (len(abc) == 1 and abc[0]["frequency"] == 10
      and pred["predictions"] and pred["predictions"][0]["next"] == "A"
      and abs(pred["predictions"][0]["probability"] - 1.0) < 1e-9)
check("1 cycle ABCx10", ok,
      f"[A->B->C] freq={abc[0]['frequency'] if abc else '?'} (expect 10); "
      f"P(A|C)={pred['predictions'][0]['probability'] if pred['predictions'] else '?'} ({dt1:.2f}s)")

# 2. Level regimes ---------------------------------------------------
d = [5.0 + (i % 4) * 0.1 for i in range(40)] + [50.0 - (i % 4) * 0.1 for i in range(40)]
disc, dt = t(Nexora().discover, d)
regs = sorted([p for p in disc["patterns"] if p["type"] == "regime"],
              key=lambda p: sum(p["sequence"]) / len(p["sequence"]))
means = [sum(p["sequence"]) / len(p["sequence"]) for p in regs]
ok = len(regs) == 2 and means[0] < 10.0 and means[1] > 40.0
check("2 two-level regimes", ok, f"regime means={[round(m, 1) for m in means]} (expect ~5/~50) ({dt:.2f}s)")

# 3. Spike anomaly ---------------------------------------------------
d = [round(10 + (i % 5) * 0.2, 1) for i in range(60)] + [100.0]
out, dt = t(Nexora().find_anomalies, d)
stat = [a for a in out["anomalies"] if a["kind"] == "statistical"]
ok = any(a["index"] == 60 for a in stat)
z = next((a["z"] for a in stat if a["index"] == 60), None)
check("3 spike anomaly", ok, f"flagged index 60 with z={z:.1f} (expect |z|>>3) ({dt:.2f}s)" if z else "not flagged")

# 4. Multivariate anomaly --------------------------------------------
pts = [[20 + (i % 4) * 0.2, 1013 + (i % 3) * 0.3] for i in range(25)] + [[35.0, 990.0]]
mv, dt = t(detect_multivariate, pts, threshold=0.8)
idx = [a["index"] for a in mv["anomalies"]]
ok = idx == [25]
s = mv["anomalies"][0]["score"] if mv["anomalies"] else 0
check("4 joint temp/pressure outlier", ok, f"flagged={idx} (expect [25]), score={s:.3f} ({dt:.2f}s)")

# 5. Seasonality ------------------------------------------------------
sig = [math.sin(2 * math.pi * i / 12) for i in range(48)]
est, dt = t(estimate_period, sig)
disc5 = Nexora().discover(sig)
seas = [p for p in disc5["patterns"] if p["type"] == "seasonal"]
ok = est["period"] == 12 and seas and seas[0]["features"]["seasonal_strength"] > 0.8
check("5 period-12 seasonality", ok,
      f"period={est['period']} strength={seas[0]['features']['seasonal_strength']:.3f} (expect 12, >0.8) ({dt:.2f}s)")

# 6. Correlation ------------------------------------------------------
rng = random.Random(42)
xs = [float(i) for i in range(20)]
ys = [3 * x + rng.uniform(-0.5, 0.5) for x in xs]
pats, dt = t(find_correlation_patterns, {"x": xs, "y": ys}, threshold=0.7)
ok = len(pats) == 1 and pats[0]["features"]["r"] > 0.95
check("6 linear correlation", ok, f"r={pats[0]['features']['r']:.4f} (expect >0.95) ({dt:.2f}s)" if pats else "none found")

# 7. Change point -----------------------------------------------------
cps, dt = t(change_points, [1.0] * 20 + [10.0] * 20, window=5, threshold_z=2.0)
ok = any(15 <= c["index"] <= 25 for c in cps)
check("7 step change point", ok, f"change at {[c['index'] for c in cps]} (expect ~20) ({dt:.2f}s)")

# 8. Markov chain -----------------------------------------------------
d = ["s1", "s2", "s3"] * 5
pr, dt = t(Nexora().predict, d)
top = pr["predictions"][0] if pr["predictions"] else {}
ctx = pr["context"][0] if pr["context"] else {}
ok = top.get("next") == "s1" and abs(top.get("probability", 0) - 1.0) < 1e-9 and ctx.get("next") == "s1"
check("8 s1->s2->s3 chain", ok,
      f"P(s1|s3)={top.get('probability')} order-1; backoff order={ctx.get('order')} -> {ctx.get('next')} ({dt:.2f}s)")

# 9. Noisy cycle ------------------------------------------------------
rng = random.Random(7)
d = list("ABC" * 8)
for j in rng.sample(range(24), 5):
    d[j] = "X"
pats, dt = t(find_frequent_sequences, d, max_n=3, min_support=2)
abc = [p for p in pats if tuple(p["sequence"]) == ("A", "B", "C")]
ok = len(abc) == 1 and abc[0]["features"]["count"] >= 3
check("9 noisy cycle (5/24 corrupted)", ok,
      f"[A->B->C] count={abc[0]['features']['count'] if abc else 0} (expect >=3) ({dt:.2f}s)")

# 10. Random negative control -----------------------------------------
rng = random.Random(42)
d = [str(rng.randint(0, 999)) for _ in range(100)]
pats, dt = t(find_frequent_sequences, d, max_n=3, min_support=3)
nx10 = Nexora()
det = nx10.detect(d[:30])
q = nx10.quality(d[:30])
ok = pats == [] and isinstance(det["explanation"], str) and 0.0 <= q["quality"] <= 1.0
check("10 random tokens (control)", ok, f"frequent seqs={len(pats)} (expect 0); no crash; quality={q['quality']:.2f} ({dt:.2f}s)")

n_ok = sum(1 for _, ok, _ in results if ok)
print(f"\nSCORE: {n_ok}/10")
sys.exit(0 if n_ok == 10 else 1)
