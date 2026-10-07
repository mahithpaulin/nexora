# Nexora — non-neural pattern-recognition engine

Stdlib-only Python (>=3.10). No neural networks, no numpy, no ML
libraries, no network. Every result carries a human-readable
explanation with cited numbers, and every result envelope carries an
explicit status: `FOUND`, `NONE`, `INSUFFICIENT_DATA`, or
`LOW_CONFIDENCE` (with minimum-data thresholds in config, so nothing
is fabricated from too little data).

```python
from nexora import Nexora
nx = Nexora()
disc = nx.discover(list("ABCABCABC"))   # closed patterns, significance-vetted
print(disc["status"], disc["explanation"])
print(nx.predict(list("ABCABCABC"))["predictions"])   # calibrated + abstains
print(nx.find_anomalies([10.0] * 30 + [25.0])["anomalies"])  # one record/event
print(nx.report(nx.detect(list("ABCABC"))))  # markdown report
nx.save("state.json"); nx.load("state.json")  # schema-versioned persistence
nx.update([1.0, 2.0, 3.0])  # true incremental streaming, bounded memory
```

```bash
pip install -e .          # or just run from the repo root (stdlib only)
python -m pytest tests/ -q
python examples/demo.py           # core story
python examples/demo_phase2.py    # regimes, correlation, seasonality
python examples/eval_10.py        # 10 ground-truth problems
python bench/run_bench.py         # evaluation harness (floors enforced)
python bench/scaling.py           # scaling benchmark (slope tripwire)
```

API: `detect/discover/match/find_anomalies/predict/get_pattern/
get_history/explain` + `quality/report/save/load/batch` +
`update/stream_predict/predict_next`.
Config is validated fail-fast (`InvalidConfigError`, a `ValueError`).
Details: `pyproject.toml` (`requires-python = >=3.10`, no dependencies).
Honest limits: `docs/LIMITATIONS.md`. Changes: `CHANGELOG.md`.
License: MIT (see `LICENSE`).

What v2 adds over v1: closed-pattern pruning (`show_all=True` to
opt out), permutation significance vetting (p-value/lift per
pattern), robust rolling median/MAD anomaly scores + level shifts +
severity, result statuses, Wilson-calibrated prediction with
abstention, bounded incremental streaming with online change
detection, single-pass mining, and the `bench/` harness with
recorded floors. Old save files still load (migrated automatically).

What v3 changes over v2 (v3.1.0 — API unchanged, 100 iterations):
any iterable works wherever a list is shown (tuple/range/generator are
iterated element-wise, not swallowed as one observation);
`predict(data, current=X)` conditions both Markov and context backoff on
`X` (evidence cites `context_query`/`context_from`); `update()` history
rows carry global `index == stream_pos` across calls. See `CHANGELOG.md`.

Architecture: `ingestion → preprocessing (observation) → features
(statistical/temporal/sequence/structural/correlation/seasonality/PCA)
→ discovery (frequency/sequences/change-points/clustering/arithmetic/
significance/pruning) → representation (Pattern) → matching
(distance/similarity/DTW) → memory (repository/lifecycle/evolution/
relationships) → scoring → prediction (Markov/context-backoff/
calibration) → explanation → API (Nexora)`. Streaming stats
(`nexora/streaming.py`), anomaly detectors (`nexora/anomaly/`), and
the perf cache (`nexora/perf/`) are stdlib-only and deterministic
(seeded where randomness is needed).
