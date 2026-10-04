"""Nexora engine tying memory + anomaly + prediction + explanation.

All sibling discovery/feature/matching imports are guarded
(try/except ImportError -> None); engine falls back to deterministic
stdlib logic when a module is missing.
"""
import collections
import statistics

try:
    from nexora.core.observation import normalize_observations, validate_observations
    from nexora.core.scoring import score_pattern
except ImportError:
    normalize_observations = validate_observations = score_pattern = None
try:
    from nexora.features.statistical import describe_series
    from nexora.features.temporal import moving_average, autocorrelation, detect_trend, fft_periodicity
    from nexora.features.sequence import ngrams, transition_counts, transition_probs
except ImportError:
    describe_series = moving_average = autocorrelation = detect_trend = None
    fft_periodicity = ngrams = transition_counts = transition_probs = None
try:
    from nexora.discovery.frequency import find_recurring_values
    from nexora.discovery.sequences import find_frequent_sequences
    from nexora.discovery.change_points import change_points
except ImportError:
    find_recurring_values = find_frequent_sequences = change_points = None
try:
    from nexora.matching.similarity import sequence_similarity, pearson_similarity, cosine_similarity
    from nexora.matching.distance import euclidean, normalized_similarity
    from nexora.matching.dtw import dtw_distance
except ImportError:
    sequence_similarity = pearson_similarity = cosine_similarity = None
    euclidean = normalized_similarity = dtw_distance = None
try:
    from nexora.discovery.clustering import find_regimes
    from nexora.discovery.clustering import windows as _embed_windows
    from nexora.features.correlation import find_correlation_patterns
    from nexora.features.seasonality import decompose, estimate_period
    from nexora.features.pca import pca as _pca
    from nexora.prediction.context import (build_context_model,
                                           predict_with_context,
                                           sequence_log_loss)
    from nexora.anomaly.multivariate import detect_multivariate
    from nexora.memory.evolution import snapshot as _evo_snapshot
    from nexora.memory.evolution import drift_score as _evo_drift
    from nexora.memory.relationships import (attach_relationships,
                                             build_relationships)
except ImportError:
    find_regimes = _embed_windows = None
    find_correlation_patterns = decompose = estimate_period = None
    _pca = build_context_model = predict_with_context = sequence_log_loss = None
    detect_multivariate = None
    _evo_snapshot = _evo_drift = None
    build_relationships = attach_relationships = None
try:
    from nexora.ingestion.quality import quality_report as _quality_report
except ImportError:
    _quality_report = None
try:
    from nexora.core.config import validate_config as _validate_config
except ImportError:
    _validate_config = None
try:
    from nexora.memory.store import (load_engine_state as _load_state,
                                     save_engine_state as _save_state)
except ImportError:
    _load_state = _save_state = None
try:
    from nexora.explanation.report import (pattern_card as _pattern_card,
                                           render_markdown as _render_markdown,
                                           render_text as _render_text)
except ImportError:
    _pattern_card = _render_markdown = _render_text = None
try:
    from nexora.api.batch import batch_process as _batch_process
    from nexora.api.batch import compare_signatures as _compare_sigs
    from nexora.api.batch import summarize_batch as _summarize_batch
except ImportError:
    _batch_process = _compare_sigs = _summarize_batch = None
try:
    from nexora.memory.repository import PatternRepository
    from nexora.memory.lifecycle import advance as _lc_advance
    from nexora.anomaly.detector import detect as _anomaly_detect
    from nexora.prediction.markov import build_transition_matrix, predict_next
    from nexora.explanation.explainer import explain_match, summarize_result
except ImportError:
    try:
        from ..memory.repository import PatternRepository
        from ..memory.lifecycle import advance as _lc_advance
        from ..anomaly.detector import detect as _anomaly_detect
        from ..prediction.markov import build_transition_matrix, predict_next
        from ..explanation.explainer import explain_match, summarize_result
    except ImportError:
        PatternRepository = _lc_advance = _anomaly_detect = None
        build_transition_matrix = predict_next = None
        explain_match = summarize_result = None

DEFAULT_CONFIG = {"z_threshold": 3.0, "min_support": 3, "max_n": 3, "window": 20, "weights": {},
                  "regimes": True, "regime_size": 8, "n_clusters": 2,
                  "correlation": True, "corr_threshold": 0.7,
                  "seasonality": True, "context_order": 2,
                  "multivariate": True, "mv_window": 5, "mv_threshold": 0.8,
                  "evolve_drift": 0.4}


def _numeric_columns(data):
    """Collect numeric raw-dict fields into {key: [numbers/None]} columns.

    Only list/tuple inputs whose items are dicts are considered; a key
    is kept when it holds at least 2 valid (int/float, non-bool)
    values. Missing/None preserved as None for pairwise deletion.
    """
    cols = {}
    if not isinstance(data, (list, tuple)):
        return {}
    raws = [d for d in data if isinstance(d, dict)]
    if not raws:
        return {}
    keys = set()
    for d in raws:
        keys.update(d.keys())
    for k in sorted(keys, key=str):
        col, valid = [], 0
        for d in raws:
            v = d.get(k)
            if isinstance(v, bool):
                col.append(None)
            elif isinstance(v, (int, float)):
                col.append(float(v))
                valid += 1
            else:
                col.append(None)
        if valid >= 2:
            cols[str(k)] = col
    return cols


def _get(p, key, default=None):
    return p.get(key, default) if isinstance(p, dict) else getattr(p, key, default)


def _rows(data):
    if normalize_observations is not None:
        try:
            r = normalize_observations(data)
            if r:
                return list(r)
        except Exception:
            pass
    seq = data if isinstance(data, list) else [data]
    out = []
    for i, item in enumerate(seq):
        if isinstance(item, dict):
            out.append({"index": item.get("index", i), "value": item.get("value", item.get("label")),
                        "label": str(item.get("label", item.get("value", ""))),
                        "timestamp": item.get("timestamp", i), "raw": item.get("raw", item)})
        else:
            out.append({"index": i, "value": item, "label": str(item), "timestamp": i, "raw": item})
    return out


def _stats(values):
    if describe_series is not None:
        try:
            return dict(describe_series(values))
        except Exception:
            pass
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return {"mean": 0.0, "stdev": 0.0, "count": 0}
    return {"mean": statistics.fmean(nums), "stdev": statistics.pstdev(nums) if len(nums) > 1 else 0.0, "count": len(nums)}


def _sim(row, pattern):
    """Similarity 0..1 of one row vs one pattern (sequence-based, deterministic)."""
    seq = [str(x) for x in (_get(pattern, "sequence", []) or [])]
    lab = str(row.get("label", row.get("value", "")))
    if sequence_similarity is not None and seq:
        try:
            score, _reason = sequence_similarity([lab], seq)
            return max(0.0, min(1.0, float(score)))
        except Exception:
            pass
    feats = _get(pattern, "features", {}) or {}
    if row.get("value") == feats.get("value"):
        return 1.0
    if lab in seq or str(row.get("value", "")) in seq:
        return 0.8
    return 0.0


class Nexora:
    """Stable public API: detect, discover, match, find_anomalies, predict,
    get_pattern, get_history, explain + quality, report, save, load, batch.
    Internal algorithms may change. Config is validated fail-fast
    (InvalidConfigError, a ValueError) when nexora.core.config exists."""

    def __init__(self, config=None):
        if _validate_config is not None:
            self.config = _validate_config(config)
        else:
            self.config = dict(DEFAULT_CONFIG)
            if config:
                self.config.update(config)
        if PatternRepository is None:
            raise ImportError("nexora.memory.repository is required")
        self.repo = PatternRepository()
        self._trail = {}
        self._snaps = {}
        self._matrix = None
        self._ctx = None
        self._lc_cfg = {"confirm_threshold": 3, "establish_freq": 10, "establish_confidence": 0.8,
                        "stale_after": 100, "retired_after": 200, "drift_threshold": 0.5}

    def _store(self, p):
        pid = self.repo.add(p)
        stored = self.repo.get(pid)
        if _lc_advance is not None and isinstance(stored, dict):
            try:
                new_state, _ = _lc_advance(dict(stored), True, self._lc_cfg)
                if new_state != stored.get("state"):
                    try:
                        self.repo.update(pid, {"state": new_state})
                    except Exception:
                        pass
                    stored["state"] = new_state
            except Exception:
                pass
        if _evo_snapshot is not None and isinstance(stored, dict):
            try:
                new_snap = _evo_snapshot(stored)
                old_snap = self._snaps.get(pid)
                if old_snap is not None and _evo_drift is not None:
                    d = _evo_drift(old_snap, new_snap)
                    if d.get("score", 0.0) >= float(self.config.get("evolve_drift", 0.4)):
                        try:
                            self.repo.update(pid, {"state": "EVOLVING"})
                        except Exception:
                            pass
                        stored["state"] = "EVOLVING"
                self._snaps[pid] = new_snap
            except Exception:
                pass
        c = stored.get("confidence", 0.5) if isinstance(stored, dict) else 0.5
        self._trail.setdefault(pid, []).append(c)
        return pid

    def discover(self, data):
        """Find recurring values + frequent sequences; store them; return evidence."""
        rows = _rows(data)
        values = [r["value"] for r in rows]
        labels = [r["label"] for r in rows]
        st = _stats(values)
        min_sup = int(self.config.get("min_support", 3))
        found = []
        if find_recurring_values is not None:
            try:
                found = list(find_recurring_values(values, min_sup) or [])
            except Exception:
                found = []
        if not found:
            cnt = collections.Counter(str(v) for v in values)
            idxs = {}
            for r in rows:
                idxs.setdefault(str(r["value"]), []).append(r["index"])
            for val, c in cnt.items():
                if c >= min_sup:
                    found.append({"value": val, "count": c, "indices": idxs[val]})
        for item in found:
            if isinstance(item, dict) and "features" in item:
                feats = item.get("features", {}) or {}
                v = feats.get("value")
                try:
                    c = int(feats.get("count", item.get("frequency", 1)) or 1)
                except (TypeError, ValueError):
                    c = 1
                occ = list(item.get("occurrences", []) or [])
            else:  # Counter-fallback shape
                v = item.get("value") if isinstance(item, dict) else item
                c = int(item.get("count", 1)) if isinstance(item, dict) else 1
                occ = list(item.get("indices", [])) if isinstance(item, dict) else []
            self._store({"type": "recurring_value", "features": {"value": v, "count": c}, "sequence": [v],
                         "relationships": {}, "frequency": c, "first_seen": occ[0] if occ else None,
                         "last_seen": occ[-1] if occ else None, "occurrences": occ,
                         "confidence": min(1.0, 0.4 + 0.1 * c), "similarity": 1.0, "novelty": 0.0,
                         "context": {"stats": st}, "metadata": {}, "state": "NEW"})
        seqs = []
        if find_frequent_sequences is not None:
            try:
                seqs = find_frequent_sequences(labels, int(self.config.get("max_n", 3)), min_sup) or []
            except Exception:
                seqs = []
        if not seqs and len(labels) > 1:
            for (a, b), c in collections.Counter(zip(labels, labels[1:])).items():
                if c >= min_sup:
                    seqs.append({"sequence": [a, b], "support": c})
        for s in seqs:
            sq = list(s.get("sequence", [])) if isinstance(s, dict) else list(s)
            if isinstance(s, dict):
                feats = s.get("features", {}) or {}
                try:
                    sup = int(feats.get("count", s.get("frequency", min_sup)) or min_sup)
                except (TypeError, ValueError):
                    sup = min_sup
                occ = list(s.get("occurrences", []) or [])
            else:
                sup, occ = min_sup, []
            self._store({"type": "frequent_sequence", "features": {"sequence": sq, "support": sup}, "sequence": sq,
                         "relationships": {}, "frequency": sup, "first_seen": occ[0] if occ else None,
                         "last_seen": occ[-1] if occ else None, "occurrences": occ,
                         "confidence": min(1.0, 0.4 + 0.1 * sup),
                         "similarity": 1.0, "novelty": 0.0, "context": {}, "metadata": {}, "state": "NEW"})
        nums = [v for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if self.config.get("regimes", True) and find_regimes is not None and len(nums) >= 16:
            try:
                rsize = max(2, int(self.config.get("regime_size", 8)))
                rk = max(2, int(self.config.get("n_clusters", 2)))
                for rp in find_regimes(nums, size=rsize, k=rk) or []:
                    feats = rp.get("features", {}) or {}
                    occ = list(rp.get("occurrences", []) or [])
                    cnt = int(feats.get("count", rp.get("frequency", 1)) or 1)
                    self._store({"type": "regime",
                                 "features": {"method": feats.get("method", "kmeans"),
                                              "cluster": feats.get("cluster"),
                                              "window": rsize, "count": cnt,
                                              "support": feats.get("support", 0.0),
                                              "centroid": [round(float(x), 4) for x in (feats.get("centroid") or [])]},
                                 "sequence": [round(float(x), 4) for x in (rp.get("sequence") or [])],
                                 "relationships": {}, "frequency": cnt,
                                 "first_seen": occ[0] if occ else None,
                                 "last_seen": occ[-1] if occ else None, "occurrences": occ,
                                 "confidence": min(1.0, 0.4 + 0.1 * cnt),
                                 "similarity": 1.0, "novelty": 0.0,
                                 "context": {"stats": st}, "metadata": {}, "state": "NEW"})
            except Exception:
                pass
        if self.config.get("correlation", True) and find_correlation_patterns is not None:
            try:
                cols = _numeric_columns(data)
                if len(cols) >= 2:
                    cps = find_correlation_patterns(cols, threshold=float(self.config.get("corr_threshold", 0.7))) or []
                    for cp in cps:
                        feats = cp.get("features", {}) or {}
                        n = int(feats.get("n", cp.get("frequency", 0)) or 0)
                        self._store({"type": "correlation",
                                     "features": {"a": feats.get("a"), "b": feats.get("b"),
                                                  "r": feats.get("r"), "strength": feats.get("strength"),
                                                  "n": n},
                                     "sequence": list(cp.get("sequence") or []),
                                     "relationships": {}, "frequency": n,
                                     "first_seen": None, "last_seen": None, "occurrences": [],
                                     "confidence": min(1.0, abs(float(feats.get("r", 0.0) or 0.0))),
                                     "similarity": 1.0, "novelty": 0.0,
                                     "context": {}, "metadata": {}, "state": "NEW"})
            except Exception:
                pass
        if self.config.get("seasonality", True) and estimate_period is not None and len(nums) >= 12:
            try:
                est = estimate_period(nums)
                if est and est.get("period"):
                    dec = decompose(nums, int(est["period"]))
                    sig = []
                    for i in range(int(est["period"])):
                        v = dec["seasonal"][i]
                        sig.append(round(float(v), 4) if v is not None else None)
                    self._store({"type": "seasonal",
                                 "features": {"period": int(est["period"]),
                                              "seasonal_strength": dec.get("seasonal_strength", 0.0),
                                              "trend_strength": dec.get("trend_strength", 0.0)},
                                 "sequence": sig,
                                 "relationships": {}, "frequency": 1,
                                 "first_seen": None, "last_seen": None, "occurrences": [],
                                 "confidence": float(dec.get("seasonal_strength", 0.0)),
                                 "similarity": 1.0, "novelty": 0.0,
                                 "context": {}, "metadata": {}, "state": "NEW"})
            except Exception:
                pass
        if build_relationships is not None:
            try:
                current = self.repo.all()
                seq_pats = [p for p in current
                            if p.get("type") in ("sequential", "frequent_sequence") and p.get("sequence")]
                if seq_pats:
                    rels = build_relationships(labels, seq_pats)
                    attached = attach_relationships([dict(p) for p in seq_pats], rels)
                    for p in attached:
                        try:
                            self.repo.update(p["id"], {"relationships": p.get("relationships", {})})
                        except Exception:
                            pass
            except Exception:
                pass
        pats = self.repo.all()
        _mean = st.get("mean", 0.0)
        try:
            _mean = float(_mean)
        except (TypeError, ValueError):
            _mean = 0.0
        expl = "Discovered %d pattern(s) from %d observation(s) (mean %.3f)." % (len(pats), len(rows), _mean)
        return {"patterns": pats, "count": len(pats), "reason": expl, "explanation": expl, "evidence": {"stats": st}}

    def match(self, observation):
        """Rank all stored patterns by similarity to one observation."""
        row = _rows([observation])[0]
        res = []
        for p in self.repo.all():
            s = _sim(row, p)
            feats = list((p.get("features", {}) or {}).keys())
            ev = explain_match(row, p.get("id"), s, feats) if explain_match else "similarity %.3f" % s
            res.append({"pattern_id": p.get("id"), "similarity": s, "matched": s >= 0.5, "evidence": ev, "explanation": ev})
        res.sort(key=lambda d: (-d["similarity"], str(d["pattern_id"])))
        return res

    def detect(self, data):
        """discover + match in one pass over the data."""
        disc = self.discover(data)
        matches = [m for r in _rows(data) for m in self.match(r)]
        expl = "detect: %d pattern(s), %d match(es) over %d observation(s)." % (disc.get("count", 0), len(matches), len(_rows(data)))
        return {"patterns": disc.get("patterns", []), "matches": matches, "reason": expl, "explanation": expl, "evidence": disc.get("evidence", {})}

    def find_anomalies(self, data):
        """Flag statistical outliers + novel/missing sequence transitions."""
        rows = _rows(data)
        st = _stats([r["value"] for r in rows])
        zt = float(self.config.get("z_threshold", 3.0))
        if _anomaly_detect is not None:
            out = _anomaly_detect(rows, st, zt, self.repo.all())
        else:
            out = [{"index": r["index"], "value": r["value"], "kind": "statistical", "score": 0.9,
                    "causes": ["z-threshold"], "explanation": "outlier"} for r in rows
                   if isinstance(r["value"], (int, float)) and st["stdev"] > 0 and abs((r["value"] - st["mean"]) / st["stdev"]) >= zt]
        if self.config.get("multivariate", True) and detect_multivariate is not None and _embed_windows is not None:
            try:
                pairs = [(r["index"], r["value"]) for r in rows
                         if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)]
                msize = max(2, int(self.config.get("mv_window", 5)))
                if len(pairs) >= 2 * msize:
                    wvecs, wstarts = _embed_windows([v for _, v in pairs], msize)
                    if wvecs:
                        mv = detect_multivariate(wvecs, threshold=float(self.config.get("mv_threshold", 0.9)))
                        for a in mv.get("anomalies", []):
                            ri = pairs[wstarts[a["index"]]][0]
                            rv = pairs[wstarts[a["index"]]][1]
                            dims = mv.get("dims", msize)
                            out.append({
                                "index": ri, "value": rv, "z": 0.0, "score": a["score"],
                                "kind": "multivariate",
                                "causes": ["window d2=%.2f over %d dims" % (a["d2"], dims)],
                                "explanation": ("Multivariate outlier at window starting index %s: "
                                                "Mahalanobis d2=%.2f across %d dims, score %.2f "
                                                "(threshold %s)." % (ri, a["d2"], dims, a["score"],
                                                                      self.config.get("mv_threshold", 0.9))),
                            })
            except Exception:
                pass
        out.sort(key=lambda d: (d.get("index", 0), d.get("kind", "")))
        expl = "Found %d anomalie(s) (z_threshold=%s)." % (len(out), zt)
        return {"anomalies": out, "count": len(out), "reason": expl, "explanation": expl, "evidence": {"stats": st}}

    def predict(self, data, current=None):
        """P(next|current) from bigram counts + backoff context model.

        predictions: first-order Markov (stable v0.1 field). context:
        variable-order backoff predictions with the order actually used.
        log_loss: mean base-2 NLL of the data under the context model
        (lower = more predictable; None when not computable).
        """
        labels = [r["label"] for r in _rows(data)]
        if build_transition_matrix is not None:
            self._matrix = build_transition_matrix(labels)
            cur = current if current is not None else (labels[-1] if labels else None)
            preds = predict_next(cur, self._matrix, top_k=3) if cur is not None else []
        else:
            cur, preds = current, []
        ctx_preds, log_loss = [], None
        if build_context_model is not None and labels:
            try:
                mo = max(0, int(self.config.get("context_order", 2)))
                clean = [l for l in labels if l is not None]
                self._ctx = build_context_model(clean, mo)
                tail = clean[-mo:] if mo > 0 else []
                ctx_preds = predict_with_context(self._ctx, tail, top_k=3) or []
                if sequence_log_loss is not None:
                    try:
                        log_loss = sequence_log_loss(self._ctx, clean)
                    except Exception:
                        log_loss = None
            except Exception:
                pass
        expl = "Predicted %d candidate(s) after '%s'." % (len(preds), cur)
        states = (self._matrix or {}).get("states", [])
        return {"current": cur, "predictions": preds, "context": ctx_preds, "log_loss": log_loss,
                "reason": expl, "explanation": expl, "evidence": {"matrix_states": states}}

    def get_pattern(self, pid):
        """Return one stored pattern dict (or None)."""
        return self.repo.get(pid)

    def get_history(self, pid):
        """Occurrences + confidence trail for one pattern."""
        p = self.repo.get(pid)
        if p is None:
            return {"pattern_id": pid, "occurrences": [], "confidence_trail": [], "reason": "Pattern not found."}
        occ = p.get("occurrences", []) if isinstance(p, dict) else _get(p, "occurrences", [])
        trail = list(self._trail.get(pid, [p.get("confidence", 0.5) if isinstance(p, dict) else 0.5]))
        return {"pattern_id": pid, "occurrences": occ, "confidence_trail": trail,
                "reason": "History for %s: %d occurrence(s)." % (pid, len(occ))}

    def explain(self, result):
        """Human-readable summary of any result dict."""
        if summarize_result is not None:
            try:
                return summarize_result(result)
            except Exception:
                pass
        return str(result)

    def quality(self, data):
        """Data-quality report for raw input (never raises on ordinary data).

        Raises TypeError for non-list/tuple/None input (programmer error).
        """
        if _quality_report is None:
            raise ImportError("nexora.ingestion.quality is required")
        return _quality_report(data)

    def report(self, result, fmt="markdown"):
        """Full human-readable report: fmt="markdown" (default) or "text".

        Never raises on malformed results (coerces with placeholders).
        """
        if fmt == "text":
            if _render_text is None:
                raise ImportError("nexora.explanation.report is required")
            return _render_text(result)
        if _render_markdown is None:
            raise ImportError("nexora.explanation.report is required")
        return _render_markdown(result)

    def save(self, path):
        """Persist engine state (config, patterns, trails) to path (atomic write).

        Returns path. Raises ImportError if store module missing, OSError
        on real IO failures.
        """
        if _save_state is None:
            raise ImportError("nexora.memory.store is required")
        return _save_state(path, self.repo, trails=self._trail, config=self.config)

    def load(self, path):
        """Load state saved by save(); replaces memory, trails, config.

        Returns {"patterns", "version", "reason"}. Raises
        FileNotFoundError/ValueError on bad files (fail-fast, documented).
        """
        if _load_state is None:
            raise ImportError("nexora.memory.store is required")
        st = _load_state(path)
        fresh = PatternRepository()
        fresh.import_patterns(st.get("patterns", []))
        self.repo = fresh
        trails = st.get("trails", {})
        self._trail = {str(k): list(v) for k, v in trails.items()} if isinstance(trails, dict) else {}
        self._snaps = {}
        cfg = st.get("config", {})
        if cfg and _validate_config is not None:
            try:
                self.config = _validate_config(cfg)
            except Exception:
                pass
        return {"patterns": self.repo.size(), "version": st.get("version"),
                "reason": "Loaded %d pattern(s) (schema v%s)." % (self.repo.size(), st.get("version"))}

    def batch(self, datasets):
        """Isolated detect() per dataset (fresh engine each); returns outcomes.

        One bad dataset records {"error": ...} and never kills the batch.
        See nexora.api.batch.summarize_batch / compare_signatures for
        rollups and diffs.
        """
        if _batch_process is None:
            raise ImportError("nexora.api.batch is required")
        return _batch_process(datasets, self.config)
