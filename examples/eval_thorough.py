"""Nexora thorough evaluation: 30 ground-truth + unknown + robustness checks.

Run: python examples/eval_thorough.py   (from repo root; stdlib only)
Exit 0 iff 30/30 pass. Deterministic. Prints per-check timing.
"""
import math
import random
import sys
import time

sys.path.insert(0, ".")

from nexora import Nexora
from nexora.anomaly.multivariate import detect_multivariate
from nexora.discovery.arithmetic import analyze_numeric_sequence
from nexora.discovery.change_points import change_points
from nexora.discovery.sequences import find_frequent_sequences
from nexora.features.correlation import find_correlation_patterns
from nexora.features.seasonality import estimate_period
from nexora.streaming import stream_anomalies

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def t(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


# 1-3 known discovery
d = list("ABC" * 10)
nx = Nexora(config={"min_support": 2, "max_n": 3})
disc, dt = t(nx.discover, d)
abc = [p for p in disc["patterns"] if tuple(p["sequence"]) == ("A", "B", "C")]
check("1 cycle ABCx10", len(abc) == 1 and abc[0]["frequency"] == 10,
      f"freq={abc[0]['frequency'] if abc else '?'} ({dt:.2f}s)")

d = [5.0 + (i % 4) * 0.1 for i in range(40)] + [50.0 - (i % 4) * 0.1 for i in range(40)]
disc, dt = t(Nexora().discover, d)
regs = sorted([p for p in disc["patterns"] if p["type"] == "regime"],
              key=lambda p: sum(p["sequence"]) / len(p["sequence"]))
means = [sum(p["sequence"]) / len(p["sequence"]) for p in regs]
check("2 two regimes", len(regs) == 2 and means[0] < 10 and means[1] > 40,
      f"means={[round(m, 1) for m in means]} ({dt:.2f}s)")

sig = [math.sin(2 * math.pi * i / 12) for i in range(48)]
est, dt = t(estimate_period, sig)
check("3 period-12", est["period"] == 12, f"period={est['period']} ({dt:.2f}s)")

# 4-6 arithmetic knowns + unknowns
def _nxt(r):
    n = r["next"]
    return n[0] if isinstance(n, list) else n


r, dt = t(analyze_numeric_sequence, [2.0, 4.0, 6.0, 8.0])
check("4 linear next=10", r is not None and abs(_nxt(r) - 10.0) < 1e-9, f"next={_nxt(r) if r else '?'} ({dt:.2f}s)")
r, dt = t(analyze_numeric_sequence, [3.0, 6.0, 12.0, 24.0])
check("5 geometric next=48", r is not None and abs(_nxt(r) - 48.0) < 1e-9, f"next={_nxt(r) if r else '?'} ({dt:.2f}s)")
r, dt = t(analyze_numeric_sequence, [1.0, 4.0, 9.0, 16.0])
check("6 quadratic next=25", r is not None and abs(_nxt(r) - 25.0) < 1e-6, f"next={_nxt(r) if r else '?'} ({dt:.2f}s)")
r, dt = t(analyze_numeric_sequence, [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
check("7 fib next=13", r is not None and abs(_nxt(r) - 13.0) < 1e-6, f"next={_nxt(r) if r else '?'} ({dt:.2f}s)")

# 8 predict_next wiring
p, dt = t(Nexora().predict_next, [2.0, 4.0, 6.0, 8.0])
check("8 predict_next=10", bool(p) and abs(p["next"] - 10.0) < 1e-9, f"next={p['next'] if p else '?'} ({dt:.2f}s)")

# 9-10 correlation known + unknown negative
rng = random.Random(42)
xs = [float(i) for i in range(20)]
ys = [3 * x + rng.uniform(-0.5, 0.5) for x in xs]
pats, dt = t(find_correlation_patterns, {"x": xs, "y": ys}, threshold=0.7)
check("9 corr r>0.95", len(pats) == 1 and pats[0]["features"]["r"] > 0.95,
      f"r={pats[0]['features']['r']:.4f}" if pats else "none")
xs2 = [float(i) for i in range(20)]
ys2 = [-2.0 * x + 5.0 for x in xs2]
pats, dt = t(find_correlation_patterns, {"x": xs2, "y": ys2}, threshold=0.7)
check("10 neg corr r<-0.99", len(pats) == 1 and pats[0]["features"]["r"] < -0.99,
      f"r={pats[0]['features']['r']:.4f}" if pats else "none")

# 11-14 anomaly knowns + unknowns
d = [round(10 + (i % 5) * 0.2, 1) for i in range(60)] + [100.0]
out, dt = t(Nexora().find_anomalies, d)
# NOTE (v2/WS4): merged records union detector kinds; `in`, not `==`.
stat = [a for a in out["anomalies"] if "statistical" in a["kind"]]
check("11 spike idx60", any(a["index"] == 60 for a in stat), f"flagged={[a['index'] for a in stat]} ({dt:.2f}s)")
d = [10.0] * 30 + [60.0] + [10.0] * 30 + [60.0]
out, dt = t(Nexora().find_anomalies, d)
idx = {a["index"] for a in out["anomalies"]}
check("12 two spikes", 30 in idx and 61 in idx, f"flagged={sorted(idx)} ({dt:.2f}s)")
out, dt = t(Nexora().find_anomalies, [7.0] * 40)
check("13 flat no false positive", [a for a in out["anomalies"] if a["kind"] == "statistical"] == [], f"n={len(out['anomalies'])} ({dt:.2f}s)")
pts = [[20 + (i % 4) * 0.2, 1013 + (i % 3) * 0.3] for i in range(25)] + [[35.0, 990.0]]
mv, dt = t(detect_multivariate, pts, threshold=0.8)
check("14 joint outlier [25]", [a["index"] for a in mv["anomalies"]] == [25], f"flagged={[a['index'] for a in mv['anomalies']]} ({dt:.2f}s)")

# 15 novel transition
nx15 = Nexora(config={"min_support": 2})
nx15.discover(list("ABCABCABCABC"))
out15, dt = t(nx15.find_anomalies, list("ABZ"))
mt = [a for a in out15["anomalies"] if a["kind"] == "missing_transition"]
check("15 novel B->Z", any(a["index"] == 2 for a in mt), f"at={[a['index'] for a in mt]} ({dt:.2f}s)")

# 16-17 change points
cps, dt = t(change_points, [1.0] * 20 + [10.0] * 20, window=5, threshold_z=2.0)
check("16 step cp ~20", any(15 <= c["index"] <= 25 for c in cps), f"at={[c['index'] for c in cps]} ({dt:.2f}s)")
cps, dt = t(change_points, [1.0] * 20 + [10.0] * 20 + [1.0] * 20, window=5, threshold_z=2.0)
check("17 return 2 cps", len(cps) >= 2, f"n={len(cps)} ({dt:.2f}s)")

# 18 noisy cycle
rng = random.Random(7)
d = list("ABC" * 8)
for j in rng.sample(range(24), 5):
    d[j] = "X"
pats, dt = t(find_frequent_sequences, d, max_n=3, min_support=2)
abc = [p for p in pats if tuple(p["sequence"]) == ("A", "B", "C")]
check("18 noisy cycle", len(abc) == 1 and abc[0]["features"]["count"] >= 3,
      f"count={abc[0]['features']['count'] if abc else 0} ({dt:.2f}s)")

# 19 random control
rng = random.Random(42)
d = [str(rng.randint(0, 999)) for _ in range(100)]
pats, dt = t(find_frequent_sequences, d, max_n=3, min_support=3)
check("19 random 0 seqs", pats == [], f"n={len(pats)} ({dt:.2f}s)")

# 20-21 robustness: empty/single/junk
nx20 = Nexora()
t0 = time.perf_counter()
ok = True
for dd in ([], [5.0], ["only"], [1.0, None, float("nan")]):
    try:
        nx20.discover(dd)
        nx20.find_anomalies(dd)
        nx20.predict(dd)
    except Exception:
        ok = False
check("20 empty/single/junk no crash", ok, f"({time.perf_counter() - t0:.2f}s)")
sout, dt = t(stream_anomalies, ["a", None, float("nan"), float("inf"), True], window=2)
check("21 stream junk []", sout == [], f"({dt:.2f}s)")

# 22 lifecycle + drift
nx22 = Nexora(config={"min_support": 2})
t0 = time.perf_counter()
nx22.discover(list("ABCABCABC"))
s1 = {p["state"] for p in nx22.repo.all()}
nx22.discover(list("ABCABCABC"))
s2 = {p["state"] for p in nx22.repo.all()}
check("22 OBSERVED->CONFIRMED", s1 == {"OBSERVED"} and "CONFIRMED" in s2,
      f"{sorted(s1)}->{sorted(s2)} ({time.perf_counter() - t0:.2f}s)")

# 23 save/load
import json as _json
import os as _os
import tempfile as _tf
nx23 = Nexora(config={"min_support": 2})
nx23.discover(list("ABCABCABC"))
t0 = time.perf_counter()
with _tf.TemporaryDirectory() as td:
    fp = _os.path.join(td, "s.json")
    nx23.save(fp)
    nx23b = Nexora()
    nx23b.load(fp)
    same = _json.dumps(nx23.repo.all(), sort_keys=True, default=str) == \
        _json.dumps(nx23b.repo.all(), sort_keys=True, default=str)
check("23 save/load fidelity", same, f"({time.perf_counter() - t0:.2f}s)")

# 24 explanation + 25 report + 26 quality
e = Nexora().discover(list("ABCABCABC"))["explanation"]
check("24 explanation", isinstance(e, str) and len(e) > 20 and any(c.isdigit() for c in e), f"len={len(e)}")
nx25 = Nexora()
md = nx25.report(nx25.detect(list("ABCABC")), fmt="markdown")
check("25 report", isinstance(md, str) and len(md) > 20, f"len={len(md)}")
q = Nexora().quality(list("ABCABC"))
check("26 quality 0..1", 0.0 <= q["quality"] <= 1.0, f"q={q['quality']:.2f}")

# 27 config fail-fast
t0 = time.perf_counter()
try:
    Nexora({"z_threshold": -1})
    ok27 = False
except ValueError:
    ok27 = True
check("27 bad config raises", ok27, f"({time.perf_counter() - t0:.2f}s)")

# 28 determinism
import json as _js
d28 = list("ABCABCABC") + [1.0, 2.0, 3.0]
t0 = time.perf_counter()
r1 = Nexora(config={"min_support": 2}).discover(d28)
r2 = Nexora(config={"min_support": 2}).discover(d28)
same28 = _js.dumps(r1["patterns"], sort_keys=True, default=str) == _js.dumps(r2["patterns"], sort_keys=True, default=str)
check("28 determinism", same28, f"({time.perf_counter() - t0:.2f}s)")

# 29 streaming jump
sout, dt = t(stream_anomalies, [5.0] * 5 + [20.0], window=5, warmup=5)
check("29 stream jump idx5", len(sout) == 1 and sout[0]["index"] == 5, f"({dt:.2f}s)")

# 30 small bench 5k
d30 = list("ABC" * 1700)[:5000]
t0 = time.perf_counter()
out30 = Nexora().discover(d30)
dt30 = time.perf_counter() - t0
check("30 bench 5k <10s", bool(out30["patterns"]) and dt30 < 10.0, f"{dt30:.2f}s patterns={len(out30['patterns'])}")

n_ok = sum(1 for _, ok, _ in results if ok)
print(f"\nSCORE: {n_ok}/30")
sys.exit(0 if n_ok == 30 else 1)
