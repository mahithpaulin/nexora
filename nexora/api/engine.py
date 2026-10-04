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

DEFAULT_CONFIG = {"z_threshold": 3.0, "min_support": 3, "max_n": 3, "window": 20, "weights": {}}


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
    get_pattern, get_history, explain. Internal algorithms may change."""

    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        if PatternRepository is None:
            raise ImportError("nexora.memory.repository is required")
        self.repo = PatternRepository()
        self._trail = {}
        self._matrix = None
        self._lc_cfg = {"confirm_threshold": 3, "establish_freq": 10, "establish_confidence": 0.8,
                        "stale_after": 100, "retired_after": 200, "drift_threshold": 0.5}

    def _store(self, p):
        pid = self.repo.add(p)
        stored = self.repo.get(pid)
        if _lc_advance is not None and isinstance(stored, dict):
            try:
                _lc_advance(stored, True, self._lc_cfg)
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
        pats = self.repo.all()
        expl = "Discovered %d pattern(s) from %d observation(s) (mean %.3f)." % (len(pats), len(rows), st.get("mean", 0.0))
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
        expl = "Found %d anomalie(s) (z_threshold=%s)." % (len(out), zt)
        return {"anomalies": out, "count": len(out), "reason": expl, "explanation": expl, "evidence": {"stats": st}}

    def predict(self, data, current=None):
        """P(next|current) from bigram counts; evidence cites observed counts."""
        labels = [r["label"] for r in _rows(data)]
        if build_transition_matrix is not None:
            self._matrix = build_transition_matrix(labels)
            cur = current if current is not None else (labels[-1] if labels else None)
            preds = predict_next(cur, self._matrix, top_k=3) if cur is not None else []
        else:
            cur, preds = current, []
        expl = "Predicted %d candidate(s) after '%s'." % (len(preds), cur)
        states = (self._matrix or {}).get("states", [])
        return {"current": cur, "predictions": preds, "reason": expl, "explanation": expl, "evidence": {"matrix_states": states}}

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
