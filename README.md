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
