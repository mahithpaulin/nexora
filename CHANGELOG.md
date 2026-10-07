# Changelog

## 3.0.0

50-iteration clarity + capability loop on top of 2.0.0 (branch `v3`,
PR #3). Public API unchanged: everything is additive or strictly
widens accepted input, so v2 code runs as-is. Each item ships with
regression tests (`tests/test_v3_*.py`) and a `docs/LIMITATIONS.md`
note where a quirk was fixed.

Inputs (I1, I10): every entry point accepts any iterable
(list/tuple/range/generator element-wise); CSV/JSON strings are
parsed, plain strings stay single observations; bare scalars are
rejected with a naming-the-fix TypeError.
Prediction (I2, I20, I39): `current=X` conditions Markov AND context
backoff (query cited in evidence); perplexity reported; per-call
`top_k`. Streaming (I3, I40): global `index == stream_pos` rows;
`stream_predict(top_k)`, `stream_stats()`, `stream_changes()`.
Read-outs (I5-I9, I35): describe/top_patterns/patterns_by_type/
pattern_card/explain_pattern/solve/state_signature.
Management (I11-I12, I22-I24): configure (unknown keys named),
reset, export/import, compare, summarize, forget, prune (I25).
Analysis (I15-I19, I30-I34, I44-I47, I49): seasonality, correlation,
regimes, change_points, frequencies, transitions, histogram, zscores,
autocorr, moving_average, trend, graph, dtw, align, reduce,
discretize. Results (I13, I38, I42-I43): min_severity filter,
z_threshold override, discover ids, quality grades. Batch (I21, I26):
detect(include_anomalies), {name: data} dict batches. Errors (I27):
strict report formats. Demo (I48): examples/demo_v3.py.

## 2.0.0

Public API unchanged (`detect/discover/match/find_anomalies/predict/
get_pattern/get_history/explain` + `quality/report/save/load/batch`,
plus additive `update/stream_predict/predict_next/discover(show_all)`).
Old save files (v1, v0) still load — migrated automatically to schema v2.

Bug fixes (each with a regression test in `tests/test_d_regress.py`):

- D1: `discover()` on categorical data no longer cites a meaningless
  `(mean 0.000)`; it reports mode, distinct-value count and entropy.
  Numeric input keeps the mean, now with its base cited.
- D2: flat series no longer flagged `novel_sequence` — the sequence
  detector skips constant series, and only label-vocabulary-relevant
  patterns are consulted. Sequence hits carry `z=None`, never a
  fabricated `0.0`.
- D3: a univariate series no longer goes through the multivariate
  detector (>= 2 numeric columns required), and same-index detector
  hits merge to one record with unioned kinds. Also fixed an
  inconsistent `mv_threshold` fallback (0.9 vs the 0.8 default).
- D4: match explanations are generated from the actual verdict —
  misses say "did not match" with the threshold cited, and only truly
  overlapping features are listed (never the pattern's full key list).
- D5: closed-pattern pruning — `ABCABCABC` yields 1 pattern, not 12
  (`show_all=True` or `prune_redundant=False` restores everything).
- D6: single accurate README; `pyproject.toml` description fixed.

New capabilities (each with tests + acceptance checks):

- WS2 pruning: closed (default) and maximal modes in
  `nexora/discovery/prune.py`; association rules subsumed by a kept
  sequence are dropped too; `evidence["pruned"]` cites each subsumption.
- WS3 significance: permutation null (seeded shuffles, shared stream:
  13.9s -> 0.5s on the noise benchmark) + analytic expectation;
  Bonferroni-corrected per-set thresholds with adaptive shuffle
  resolution; annotated (`pattern["significance"]`) always, dropped
  only with powered data (`sig_min_n`); singletons never dropped.
  Seeded uniform noise: 0 sequential false patterns.
- WS4 robust anomalies: causal rolling median/MAD (`modified_z`,
  MAD=0 jumps -> inf), seasonality-centered residuals, mean-shift
  level-shift records, severity levels on every record, one merged
  record per event. Flat series stay silent.
- WS5 statuses: `FOUND/NONE/INSUFFICIENT_DATA/LOW_CONFIDENCE` on every
  dict-result with citing reasons; minimum-data thresholds in config
  (`min_data_discover/anomalies/predict`).
- WS6 prediction: MLE probabilities unchanged, now with Wilson 95%
  intervals and evidence counts; `predict_next` abstains below
  `abstain_threshold`/`min_evidence` instead of guessing; order-2
  backoff reaches 1.00 held-out accuracy where order-1 gets 0.50.
- WS7 streaming: `update()` folds observations in bounded memory
  (Welford + sliding window + bigram counts + capped history; 100k
  observations peak 2.63MB, test bound 15MB), never reprocessing
  history (white-box tested); online change detection with global
  `stream_pos`; `stream_predict()` matches batch `predict()` exactly.
- WS8 performance: single-pass mining (O(n) per n-gram order,
  byte-identical outputs); `bench/scaling.py` log-log slope tripwire
  (all ops ~1.0, caps ~1.6; CI fails on breach).
- WS9 evaluation: `bench/` datasets with ground truth, PRF metrics,
  naive baselines, recorded floors (CI fails on drops); benchmark
  tables committed (`bench/BENCH_RESULTS.md`,
  `bench/scaling_results.json`).
- WS10 hardening: typed results (`Pattern/AnomalyRecord/Prediction/
  MatchEntry/StatusEnvelope` in `nexora/core/result.py`, hints on the
  engine API), schema-v2 save/load with v0+v1 migration, determinism
  tests, this changelog, `docs/LIMITATIONS.md`.

Test updates with written justification (the old assertions encoded
bugs D3/D5 or the superseded schema, not intended behavior):
`test_phase2::test_engine_multivariate_anomaly`,
`test_v2::test_multivariate_end_index_flagged` and
`::test_engine_structural_evidence`,
`test_v1::test_engine_quality_report_batch_prune` and
`::test_store_roundtrip_and_migration`, plus `==` -> `in` kind
filters where records merged (behavior strengthened, not weakened).

## 1.0.0

Streaming stats (Welford/SlidingStats), perf cache (LRU, cached DTW),
validated fail-fast config, data-quality report, atomic versioned
save/load with v0 migration, markdown/text reports, batch API,
file-path ingestion, closed-form arithmetic solving, thorough evals
(10 + 30 ground-truth problems). See git history for details.

## 0.1 / phase 2 (historical)

Initial stdlib-only engine (ingestion -> explanation layers) plus
clustering/PCA/correlation/seasonality/context-backoff/multivariate/
evolution/relationships modules. Superseded by 1.0.0 and 2.0.0 above.
