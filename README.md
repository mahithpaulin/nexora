# Nexora v1.0.0 — non-neural pattern-recognition engine

Stdlib-only Python. No neural networks, no numpy, no GPU. Every result
carries a human-readable explanation with cited numbers.

```python
from nexora import Nexora
nx = Nexora()
print(nx.discover(list("ABCABCABC"))["explanation"])
print(nx.predict(list("ABCABCABC"))["predictions"])
print(nx.find_anomalies([10.0] * 30 + [25.0])["anomalies"])
print(nx.report(nx.detect(list("ABCABC"))))  # markdown report
nx.save("state.json"); nx.load("state.json")  # persistence
```

```bash
pip install -e .          # or just run from the repo root (stdlib only)
python -m pytest tests/ -q
python examples/demo.py           # core story
python examples/demo_phase2.py    # regimes, correlation, seasonality
```

API: `detect/discover/match/find_anomalies/predict/get_pattern/
get_history/explain` + `quality/report/save/load/batch`.
Config is validated fail-fast (`InvalidConfigError`, a `ValueError`).
Details: `pyproject.toml` (`requires-python = >=3.10`, no dependencies).
License: MIT (see `LICENSE`).

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
