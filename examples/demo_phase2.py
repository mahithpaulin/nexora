"""Nexora Phase 2 demo: regimes, correlation, seasonality, context
prediction, multivariate anomalies, evolution, relationships.

Run: python examples/demo_phase2.py   (from repo root; stdlib only)
"""
import math
import sys

sys.path.insert(0, ".")

from nexora import Nexora
from nexora.features.pca import pca, reconstruction_error
from nexora.memory.evolution import drift_score, snapshot, track

print("=== NEXORA PHASE 2 DEMO (non-neural, stdlib-only) ===\n")

# --- 1. Regime discovery: two operating modes ---
nx = Nexora()
two_mode = [1.0 + (i % 3) * 0.05 for i in range(30)] + \
           [8.0 - (i % 3) * 0.05 for i in range(30)]
disc = nx.discover(two_mode)
print("1. REGIMES in 30x~1.0 + 30x~8.0:")
for p in [p for p in disc["patterns"] if p["type"] == "regime"]:
    c = p["features"]["centroid"]
    print(f"  {p['id']} cluster={p['features']['cluster']} "
          f"count={p['frequency']} support={p['features']['support']:.2f} "
          f"centroid_mean={sum(c) / len(c):.2f}")

# --- 2. Correlation patterns from multi-field rows ---
nx2 = Nexora()
rows = [{"temp": float(i), "pressure": float(100 - 2 * i),
         "humidity": float((i * 7) % 11)} for i in range(20)]
disc2 = nx2.discover(rows)
print("\n2. CORRELATIONS (temp vs pressure vs humidity):")
for p in [p for p in disc2["patterns"] if p["type"] == "correlation"]:
    f = p["features"]
    print(f"  {f['a']} <-> {f['b']}: r={f['r']:.3f} (n={f['n']})")

# --- 3. Seasonality: weekly cycle over 8 weeks ---
nx3 = Nexora()
weekly = [10 + 3 * math.sin(2 * math.pi * i / 7) + (i % 2) * 0.1
          for i in range(56)]
disc3 = nx3.discover(weekly)
print("\n3. SEASONALITY (period-7 sine, 56 points):")
for p in [p for p in disc3["patterns"] if p["type"] == "seasonal"]:
    f = p["features"]
    print(f"  period={f['period']} seasonal_strength={f['seasonal_strength']:.3f} "
          f"trend_strength={f['trend_strength']:.3f} conf={p['confidence']:.3f}")

# --- 4. Context (order-2) prediction + log-loss ---
seq = list("ABCABCABCABDABC")
pred = nx.predict(seq)
print("\n4. CONTEXT PREDICTION after", "".join(seq[-4:]), ":")
for pr in pred["context"]:
    print(f"  next={pr['next']!r} P={pr['probability']:.2f} order={pr['order']} | {pr['evidence']}")
print(f"  sequence log-loss: {pred['log_loss']:.3f} bits (lower = more predictable)")

# --- 5. Multivariate anomaly: joint temp/pressure outlier ---
vals = [[20 + (i % 4) * 0.2, 1013 + (i % 3) * 0.3] for i in range(25)]
vals.append([35.0, 990.0])  # hot + low pressure together
out = nx.find_anomalies([{"value": v[0]} for v in vals])  # 1-D view first
print("\n5. ANOMALIES (25 normal pairs + [35C, 990hPa]):")
print(f"  1-D detectors found {out['count']}; multivariate view:")
from nexora.anomaly.multivariate import detect_multivariate
mv = detect_multivariate(vals, threshold=0.8)
for a in mv["anomalies"]:
    print(f"  row {a['index']}: score={a['score']:.3f} d2={a['d2']:.1f} | {a['explanation']}")

# --- 6. Evolution: pattern drift over time ---
snaps = [snapshot({"id": "P-1", "type": "t", "features": {"mean": 10.0},
                           "confidence": 0.5 + 0.1 * i, "frequency": 5})
         for i in range(4)]
print("\n6. EVOLUTION:", track(snaps)["reason"])
d = drift_score(snaps[0], snaps[-1])
print(f"  total drift first->last: {d['score']:.3f} ({d['reason']})")

# --- 7. Pattern relationships: P-009 -> P-017 -> P-024 chain ---
nx4 = Nexora(config={"min_support": 1, "max_n": 2})
chain = ["P-009", "P-017", "P-024", "P-009", "P-017", "P-024"]
nx4.discover(chain)
print("\n7. RELATIONSHIPS in P-009 -> P-017 -> P-024 x2:")
for p in nx4.repo.find_by_type("frequent_sequence")[:4]:
    r = p.get("relationships", {})
    if r.get("commonly_followed_by"):
        print(f"  {p['sequence']} followed_by={r['commonly_followed_by']} "
              f"probs={ {k: round(v, 2) for k, v in r['probs']['followed_by'].items()} }")

# --- 8. PCA on regime windows ---
from nexora.discovery.clustering import windows
vecs, _ = windows(two_mode, 8)
model = pca(vecs, 2)
print(f"\n8. PCA on regime windows: ratio={ [round(x, 3) for x in model['explained_ratio']]} "
      f"recon_err={reconstruction_error(vecs, model):.4f}")
print("\nDone. Every number above is computed, cited, and reproducible (seeded).")
