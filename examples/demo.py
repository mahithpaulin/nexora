"""Nexora demo on synthetic data.

Shows: (1) discovered patterns, (2) frequencies, (3) confidence,
(4) observation similarity, (5) anomalies, (6) why each result happened.

Run: python examples/demo.py   (from repo root; stdlib only)
"""
import json
import sys

sys.path.insert(0, ".")

from nexora import Nexora

print("=== NEXORA DEMO (non-neural, stdlib-only) ===\n")

# --- 1. Repeating categorical pattern with noise + missing value ---
nx = Nexora(config={"min_support": 2, "max_n": 3, "z_threshold": 3.0})
data = list("ABCABCABCAXCABC")  # [A->B->C] x3, one corrupted repeat, then clean
data.insert(5, None)            # a missing observation
print("input:", "".join("?" if v is None else v for v in data))

disc = nx.discover([v for v in data if v is not None])
print("\n1. PATTERNS DISCOVERED:")
for p in disc["patterns"][:8]:
    print(f"  {p['id']} [{p['type']}] seq={p['sequence']} "
          f"freq={p['frequency']} conf={p['confidence']:.2f} state={p.get('state')}")
print("\n2/3. FREQUENCY + CONFIDENCE: see freq=/conf= columns above.")
print("   explanation:", disc["explanation"])

# --- 4. similarity between observations ---
print("\n4. SIMILARITY:")
for obs in ("A", "C", "Z"):
    m = nx.match(obs)[0]
    print(f"  obs {obs!r} -> {m['pattern_id']} similarity={m['similarity']:.2f} "
          f"matched={m['matched']}")
    print(f"    why: {m['explanation']}")

# --- 5/6. numeric series with outlier + trend + prediction ---
# Fresh engine: sequence checks only fire against known patterns, so the
# numeric story stays purely statistical. Baseline of 30 keeps the
# single-outlier z-score well above the 3.0 threshold.
nx2 = Nexora(config={"min_support": 2, "max_n": 3, "z_threshold": 3.0})
vals = [round(10 + (i % 5) * 0.1 - 0.2, 2) for i in range(30)] + [25.0]
print("\n5. ANOMALIES in 30x~10.0 + [25.0]:")
anom = nx2.find_anomalies(vals)
for a in anom["anomalies"]:
    _z = a.get("z")
    _zs = ("%.2f" % _z) if isinstance(_z, (int, float)) else "n/a"
    print(f"  index={a['index']} value={a['value']} kind={a['kind']} "
          f"score={a['score']:.2f} z={_zs}")
    print(f"    why: {a['explanation']}")

seq = list("ABCABCABC")
pred = nx.predict(seq)
print("\n6. PREDICTION after", "".join(seq[-3:]), ":")
for pr in pred["predictions"]:
    print(f"  next={pr['next']!r} P={pr['probability']:.2f} | {pr['evidence']}")

print("\nHISTORY of", disc["patterns"][0]["id"], ":",
      json.dumps(nx.get_history(disc["patterns"][0]["id"]),
                 default=str)[:200], "...")
print("\nOVERALL:", nx.explain(disc))
