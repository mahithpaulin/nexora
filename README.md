# Nexora — non-neural pattern-recognition engine

Stdlib-only Python (>=3.10). No neural networks, no numpy, no ML
libraries, no network. Every result carries a human-readable
explanation with cited numbers.

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
python examples/eval_10.py        # 10 ground-truth problems
python bench/run_bench.py         # evaluation harness with floors
python bench/scaling.py           # scaling benchmark (time vs n)
```

API: `detect/discover/match/find_anomalies/predict/get_pattern/
get_history/explain` + `quality/report/save/load/batch`.
Config is validated fail-fast (`InvalidConfigError`, a `ValueError`).
Details: `pyproject.toml` (`requires-python = >=3.10`, no dependencies).
Honest limits: `docs/LIMITATIONS.md`. Changes: `CHANGELOG.md`.
License: MIT (see `LICENSE`).

Architecture: `ingestion → preprocessing (observation) → features
(statistical/temporal/sequence/structural/correlation/seasonality/PCA)
→ discovery (frequency/sequences/change-points/clustering/arithmetic)
→ representation (Pattern) → matching (distance/similarity/DTW) →
memory (repository/lifecycle/evolution/relationships) → scoring →
prediction (Markov/context-backoff/transitions) → explanation → API
(Nexora)`. Streaming stats (`nexora/streaming.py`), perf cache
(`nexora/perf/`), and significance testing
(`nexora/discovery/significance.py`) are stdlib-only and deterministic
(seeded where randomness is needed).
