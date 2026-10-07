# Nexora — honest limitations

What Nexora cannot do, or does only within stated bounds. Read this
before trusting an output.

## Method limits

- Non-neural by design: no hidden states, no embeddings, no learned
  representations. It counts, compares and extrapolates — it does not
  understand. Anything needing semantics, language, images or
  nonlinearity beyond short closed forms is out of scope.
- Singleton frequencies are factual counts, not discoveries: on uniform
  noise every token recurs ~15x and all 20 singletons are reported.
  Significance vetting applies to sequential claims only (a bare
  frequency carries no order information; its shuffle null is
  degenerate by construction).
- Small samples are labeled, not vetted: below `sig_min_n` (default 30
  labels) patterns are annotated but never significance-dropped, and
  `discover()` returns `LOW_CONFIDENCE`. Tiny inputs return
  `INSUFFICIENT_DATA` with a reason.
- v3: `predict(data, current=X)` conditions BOTH the Markov field and
  the context backoff on `X` (the backoff query ends with `X` and falls
  back to lower orders when unseen; restating the last label changes
  nothing). `evidence` cites the query (`context_query`/`context_from`).
- Wilson intervals are approximations; `abstain_threshold` (0.5) and
  `min_evidence` (2) defaults suit clean symbolic data and may need
  tuning per domain.
- Rolling MAD on fast oscillations fires repeatedly (a period-2 cycle
  looks like constant jumping to a 3-point trailing history) — pass the
  period (automatic when a qualifying seasonal pattern is stored) for
  residual scoring.
- Merged anomaly records keep the statistical z only; other detectors
  contribute kinds, scores and causes, not z-scores.

## Input and ingestion quirks (pre-existing, preserved for compatibility)

- v3: every engine entry point (plus `parse`/`quality_report`)
  iterates any iterable element-wise (list, tuple, range, generator,
  ...). A bare scalar or row-dict is one observation, and the
  TypeError names the fix (`wrap a single value as [value]`).
- The ingestion normalizer coerces `True` -> `1.0` and labels numerics
  `None`. `None`/`NaN` labels never form patterns (they are skipped in
  mining and scoring), which is why pure-numeric discovers show no
  n-gram sequences.
- Dict rows without a `"value"` key fall back to label-derived values;
  single-column CSV rows are parsed as scalars.

## Scope boundaries

- Pattern mining (`discover`) is batch. Streaming (`update`) covers
  running stats, transitions and change events only, and stream state
  is session-only (never persisted by `save()`).
- The engine multivariate path embeds the single `value` column in
  windows (a univariate spike test, honestly gated to multi-column
  input); true joint modeling lives in `detect_multivariate` direct use.
- v3: stream rows carry global indices — history row `index` equals
  `stream_pos` (0-based over everything streamed), so change events and
  history agree across `update()` calls.
- The perf cache (`cached_dtw`, LRU) is proven for direct DTW use but
  is not wired into `match()`; `match()` ranks by sequence/numeric
  similarity without caching.
- Discover with significance vetting costs shuffles: ~3s at n=5000
  periodic on the reference box (linear slope, documented in
  `bench/scaling_results.json`).
- `match()` returns a ranked list (backward compatibility), not a
  status envelope; each entry's `matched` verdict is its status.
- Type hints cover the public engine API, result shapes and all v2
  modules; older internals carry partial hints and are annotated
  opportunistically, not exhaustively.
- Stdlib-only performance ceiling: single-threaded, no vectorization;
  100k streaming takes seconds, 5k discover with vetting takes seconds.
  That is the documented trade for zero dependencies.
