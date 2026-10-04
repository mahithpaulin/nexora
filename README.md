# Nexora

Opencode bot enabled — comment `/oc` or `/opencode` on any issue or PR to
get AI help (code, review, research). Runs on GitHub Actions with a free
Zen model (`opencode/space-bunny-free`).

## Nexora Core v0.1 — non-neural pattern-recognition engine

Stdlib-only Python. No neural networks, no numpy, no GPU. Every result
carries a human-readable explanation with cited numbers.

```
pip install -e .          # or just run from the repo root (stdlib only)
python -m pytest tests/ -q
python examples/demo.py
```

```python
from nexora import Nexora
nx = Nexora()
print(nx.discover(list("ABCABCABC"))["explanation"])
print(nx.predict(list("ABCABCABC"))["predictions"])
print(nx.find_anomalies([10.0] * 30 + [25.0])["anomalies"])
```

Layers: `ingestion → preprocessing (observation) → features
(statistical/temporal/sequence/structural) → discovery
(frequency/sequences/change-points) → representation (Pattern) →
matching (distance/similarity/DTW) → memory (repository/lifecycle) →
scoring → prediction (Markov/transitions) → explanation → API (Nexora)`.
See `pyproject.toml` (`requires-python = >=3.10`, no dependencies).

## Phase 2 — discovery & intelligence (v0.2-dev)

New modules, still stdlib-only and deterministic:

- `discovery/clustering.py` — k-means, DBSCAN, agglomerative on
  windowed embeddings → `regime` patterns wired into `discover()`
- `features/pca.py` — power-iteration PCA + projection + recon error
- `features/correlation.py` — covariance/correlation matrices,
  cross-correlation, `correlation` patterns from multi-field rows
- `features/seasonality.py` — additive decomposition + autocorr period
  estimation → `seasonal` patterns wired into `discover()`
- `prediction/context.py` — variable-order Markov with backoff +
  sequence log-loss; exposed as `predict()["context"]`
- `anomaly/multivariate.py` — Mahalanobis detector, also run on
  windowed 1-D series inside `find_anomalies()` (`kind="multivariate"`)
- `memory/evolution.py` — snapshots, drift scores, trend tracking;
  sustained drift flips patterns to `EVOLVING` (lifecycle now persists)
- `memory/relationships.py` — `commonly_preceded_by/followed_by`
  attached to sequential patterns on every `discover()`

```bash
python -m pytest tests/ -q
python examples/demo_phase2.py
```
