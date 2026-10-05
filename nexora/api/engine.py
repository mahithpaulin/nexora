"""Nexora engine tying memory + anomaly + prediction + explanation.

All sibling discovery/feature/matching imports are guarded
(try/except ImportError -> None); engine falls back to deterministic
stdlib logic when a module is missing.
"""
import collections
import os
import statistics

try:
    from nexora.ingestion.parser import parse as _parse_path
except ImportError:
    _parse_path = None

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
    from nexora.discovery.prune import closed_keep as _closed_keep
    from nexora.discovery.prune import is_contiguous_subsequence as _is_subseq
except ImportError:
    find_recurring_values = find_frequent_sequences = change_points = None
    _closed_keep = _is_subseq = None
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
    from nexora.anomaly.robust import robust_detect as _robust_detect
    from nexora.anomaly.robust import level_shift_records as _level_shifts
    from nexora.anomaly.robust import severity_of as _severity_of
except ImportError:
    _robust_detect = _level_shifts = _severity_of = None
try:
    from nexora.discovery.arithmetic import analyze_numeric_sequence as _analyze_seq
except ImportError:
    _analyze_seq = None
try:
    from nexora.features.structural import (cooccurrence_graph as _co_graph,
                                            connected_components as _co_comps,
                                            association_rules as _assoc_rules)
except ImportError:
    _co_graph = _co_comps = _assoc_rules = None
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
                  "prune_redundant": True,
                  "regimes": True, "regime_size": 8, "n_clusters": 2,
                  "correlation": True, "corr_threshold": 0.7,
                  "seasonality": True, "context_order": 2,
                  "multivariate": True, "mv_window": 5, "mv_threshold": 0.8,
                  "robust": True, "robust_window": 20, "robust_threshold": 3.5,
                  "level_shifts": True, "ls_window": 10, "ls_threshold": 3.0,
                  "abstain_threshold": 0.5, "min_evidence": 2,
                  "evolve_drift": 0.4, "max_period": 256}


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


def _is_file_path(data):
    return isinstance(data, str) and os.path.exists(data) and os.path.isfile(data)


def _rows(data):
    if isinstance(data, os.PathLike) and _parse_path is not None:
        r = _parse_path(data)  # FileNotFoundError propagates: explicit path must exist
        return list(r)
    if _is_file_path(data) and _parse_path is not None:
        try:
            r = _parse_path(data)
            if r:
                return list(r)
            return []
        except Exception:
            return []
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


_MATCH_THRESHOLD = 0.5


def _overlap_features(row, pattern):
    """Feature names where this row actually agrees with the pattern.

    Only genuinely overlapping evidence is returned (value equality,
    label membership in the pattern sequence) — never the pattern's
    full feature-key list. Match explanations must cite real evidence
    (D4), so this is computed from the same row/pattern pair that the
    similarity verdict was computed from.
    """
    try:
        feats = pattern.get("features", {}) if isinstance(pattern, dict) else {}
    except Exception:
        feats = {}
    if not isinstance(feats, dict):
        feats = {}
    try:
        seq = [str(x) for x in (pattern.get("sequence", []) or [])] if isinstance(pattern, dict) else []
    except Exception:
        seq = []
    try:
        lab = str(row.get("label", row.get("value", "")))
    except Exception:
        lab = ""
    hit = []
    try:
        if "value" in feats and row.get("value") == feats.get("value"):
            hit.append("value")
    except Exception:
        pass
    if lab and lab in seq:
        hit.append("sequence")
    return hit


def _merge_anomalies(out):
    """Merge detector hits so one event yields one record (D3).

    Groups by index, keeps the max score, unions kinds/causes, keeps a
    statistical z-score when one was measured (else None — never a
    fabricated 0.0), and regenerates the explanation from the merged
    decision.
    """
    groups = {}
    order = []
    for a in out:
        if not isinstance(a, dict):
            continue
        try:
            idx = a.get("index", 0)
        except Exception:
            continue
        if idx not in groups:
            groups[idx] = []
            order.append(idx)
        groups[idx].append(a)
    merged = []
    for idx in order:
        hits = groups[idx]
        if len(hits) == 1:
            merged.append(hits[0])
            continue
        kinds = sorted({str(h.get("kind", "unknown")) for h in hits})
        try:
            score = max(float(h.get("score", 0.0) or 0.0) for h in hits)
        except (TypeError, ValueError):
            score = 0.0
        z = None
        for h in hits:
            if str(h.get("kind")) == "statistical" and isinstance(h.get("z"), (int, float)):
                z = h["z"]
                break
        causes = []
        for h in hits:
            for c in h.get("causes", []) or []:
                if c not in causes:
                    causes.append(c)
        val = hits[0].get("value")
        expl = ("Merged %d detector hit(s) at index %s (value %s): kinds [%s]; "
                "top score %.2f%s; causes: %s."
                % (len(hits), idx, val, ", ".join(kinds), score,
                   (" with z=%.2f" % z) if z is not None else "",
                   "; ".join(str(c) for c in causes) or "none"))
        merged.append({"index": idx, "value": val, "z": z, "score": score,
                       "kind": "+".join(kinds), "causes": causes, "explanation": expl,
                       "severity": _severity_of(score) if _severity_of else "high"})
    merged.sort(key=lambda d: (d.get("index", 0), str(d.get("kind", ""))))
    return merged


def _sim(row, pattern):
    """Similarity 0..1 of one row vs one pattern (sequence + numeric, deterministic).

    v1 categorical path runs first and is unchanged: sequence_similarity on
    str-normalized [label] vs str-normalized sequence, then exact value
    match (1.0), then membership (0.8), else 0.0. v2 adds a numeric
    distance-to-centroid-mean path for regime/seasonal patterns (or any
    pattern with a centroid / numeric sequence) when the row value is
    numeric: d = euclidean([v], [mean]) mapped via
    normalized_similarity(d, scale) with scale from pattern features when
    present else 1.0. Returns max(sequence_score, numeric_score).
    """
    try:
        seq = [str(x) for x in (_get(pattern, "sequence", []) or [])]
    except Exception:
        seq = []
    try:
        lab = str(row.get("label", row.get("value", "")))
    except Exception:
        lab = ""
    try:
        row_val_s = str(row.get("value", ""))
    except Exception:
        row_val_s = ""
    feats = _get(pattern, "features", {}) or {}
    if not isinstance(feats, dict):
        feats = {}
    seq_score, seq_reason, seq_ran = 0.0, "no sequence", False
    if sequence_similarity is not None and seq:
        try:
            _s, _r = sequence_similarity([lab], seq)
            seq_score = max(0.0, min(1.0, float(_s)))
            seq_reason = str(_r)
            seq_ran = True
        except Exception:
            seq_ran = False
    if seq_ran:
        base = seq_score
    elif row.get("value") == feats.get("value"):
        base, seq_reason = 1.0, "exact value match"
    elif lab in seq or row_val_s in seq:
        base, seq_reason = 0.8, "label/value in sequence"
    else:
        base, seq_reason = 0.0, "no match"
    num_score, num_reason = 0.0, "n/a"
    try:
        v = row.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("non-numeric row")
        if isinstance(v, float) and v != v:
            raise ValueError("NaN row")
        fv = float(v)
        ptype = _get(pattern, "type", "")
        cent = feats.get("centroid")
        cent_nums = []
        if isinstance(cent, (list, tuple)):
            for _x in cent:
                if isinstance(_x, bool):
                    continue
                if isinstance(_x, (int, float)) and _x == _x:
                    cent_nums.append(float(_x))
        seq_nums = []
        for _x in (_get(pattern, "sequence", []) or []):
            if isinstance(_x, bool):
                continue
            if isinstance(_x, (int, float)) and _x == _x:
                seq_nums.append(float(_x))
            else:
                try:
                    seq_nums.append(float(str(_x)))
                except (TypeError, ValueError):
                    continue
        if ptype in ("regime", "seasonal") or cent_nums or seq_nums:
            ref_vals = cent_nums if cent_nums else seq_nums
            if not ref_vals:
                raise ValueError("no numeric reference")
            try:
                ref = float(statistics.fmean(ref_vals))
            except Exception:
                ref = sum(ref_vals) / len(ref_vals)
            scale = 1.0
            for _k in ("stdev", "std", "stddev", "scale", "spread"):
                try:
                    if feats.get(_k) is not None:
                        scale = float(feats.get(_k))
                        break
                except (TypeError, ValueError):
                    continue
            if not (isinstance(scale, float) and scale > 0):
                scale = 1.0
            try:
                d = float(euclidean([fv], [ref])) if euclidean is not None else abs(fv - ref)
            except Exception:
                d = abs(fv - ref)
            try:
                if normalized_similarity is not None:
                    num_score = max(0.0, min(1.0, float(normalized_similarity(d, scale=scale))))
                else:
                    num_score = 1.0 / (1.0 + d / scale)
            except Exception:
                num_score = 0.0
            num_reason = "d=%.4f scale=%.4f" % (d, scale)
    except Exception:
        num_score = 0.0
    try:
        return max(0.0, min(1.0, float(max(base, num_score))))
    except Exception:
        return base


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
        if self.config.get("scoring", True) and score_pattern is not None and isinstance(stored, dict):
            try:
                try:
                    old_c = float(stored.get("confidence", 0.5))
                except (TypeError, ValueError):
                    old_c = 0.5
                old_c = max(0.0, min(1.0, old_c))
                try:
                    freq = stored.get("frequency", 1)
                except Exception:
                    freq = 1
                try:
                    _total = 0
                    for _p in self.repo.all():
                        try:
                            _total += int(_p.get("frequency", 1) or 1)
                        except (TypeError, ValueError):
                            _total += 1
                    total_n = max(1, _total)
                except Exception:
                    total_n = 1
                try:
                    _nov = float(stored.get("novelty", 0.0) or 0.0)
                except (TypeError, ValueError):
                    _nov = 0.0
                consistency = 1.0 - max(0.0, min(1.0, _nov))
                try:
                    sim_v = float(stored.get("similarity", 1.0))
                except (TypeError, ValueError):
                    sim_v = 1.0
                try:
                    _max_last, _mine = None, stored.get("last_seen")
                    for _q in self.repo.all():
                        _ls = _q.get("last_seen")
                        if _ls is None:
                            continue
                        try:
                            if _max_last is None or _ls > _max_last:
                                _max_last = _ls
                        except TypeError:
                            continue
                    recency = 1.0 if (_mine is not None and _mine == _max_last) else 0.5
                except Exception:
                    recency = 0.5
                try:
                    rels = stored.get("relationships", {}) or {}
                    pred_s = 0.0
                    _vals = list(rels.values()) if isinstance(rels, dict) else list(rels)
                    for _rv in _vals:
                        try:
                            if isinstance(_rv, dict):
                                for _k in ("probability", "prob", "confidence", "strength", "score"):
                                    if _rv.get(_k) is not None:
                                        pred_s = max(pred_s, max(0.0, min(1.0, float(_rv[_k]))))
                            elif isinstance(_rv, (list, tuple)):
                                for _e in _rv:
                                    if isinstance(_e, dict):
                                        _e = _e.get("probability", 0.0)
                                    pred_s = max(pred_s, max(0.0, min(1.0, float(_e))))
                            elif isinstance(_rv, (int, float)) and not isinstance(_rv, bool):
                                pred_s = max(pred_s, max(0.0, min(1.0, float(_rv))))
                        except (TypeError, ValueError):
                            continue
                except Exception:
                    pred_s = 0.0
                _unc = max(0.0, min(1.0, 1.0 - old_c))
                try:
                    _w = self.config.get("weights", {})
                except Exception:
                    _w = {}
                if not isinstance(_w, dict):
                    _w = {}
                res = score_pattern(frequency=freq, total_n=total_n, consistency=consistency,
                                    similarity=sim_v, recency=recency, predictive_strength=pred_s,
                                    noise=0.0, uncertainty=_unc, weights=_w)
                try:
                    scored_c = max(0.0, min(1.0, float(res.get("confidence", old_c))))
                except (TypeError, ValueError):
                    scored_c = old_c
                blended = 0.5 * old_c + 0.5 * scored_c
                detail = dict(res) if isinstance(res, dict) else {"confidence": scored_c}
                detail["blended_from"] = old_c
                detail["total_n"] = total_n
                try:
                    self.repo.update(pid, {"confidence": blended, "score_detail": detail})
                except Exception:
                    pass
                stored["confidence"] = blended
                stored["score_detail"] = detail
            except Exception:
                pass
        c = stored.get("confidence", 0.5) if isinstance(stored, dict) else 0.5
        self._trail.setdefault(pid, []).append(c)
        return pid

    def discover(self, data, *, show_all=False):
        """Find recurring values + frequent sequences; store them; return evidence.

        Closed-pattern pruning (WS2) drops a recurring/sequential
        candidate when a longer kept sequence contains it with the same
        support-count. Pass show_all=True (or set config
        prune_redundant=False) to store every candidate.
        """
        rows = _rows(data)
        values = [r["value"] for r in rows]
        labels = [r["label"] for r in rows]
        st = _stats(values)
        min_sup = int(self.config.get("min_support", 3))
        try:
            ids_before = set(str(_p.get("id")) for _p in self.repo.all() if isinstance(_p, dict))
        except Exception:
            ids_before = set()
        _structural_ev = {}
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
        _cands = []  # (seq_key, count, store_dict); pruned below (WS2)
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
            _cands.append((((str(v),), c,
                            {"type": "recurring_value", "features": {"value": v, "count": c}, "sequence": [v],
                             "relationships": {}, "frequency": c, "first_seen": occ[0] if occ else None,
                             "last_seen": occ[-1] if occ else None, "occurrences": occ,
                             "confidence": min(1.0, 0.4 + 0.1 * c), "similarity": 1.0, "novelty": 0.0,
                             "context": {"stats": st}, "metadata": {}, "state": "NEW"})))
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
            _cands.append(((tuple(str(x) for x in sq), sup,
                            {"type": "frequent_sequence", "features": {"sequence": sq, "support": sup}, "sequence": sq,
                             "relationships": {}, "frequency": sup, "first_seen": occ[0] if occ else None,
                             "last_seen": occ[-1] if occ else None, "occurrences": occ,
                             "confidence": min(1.0, 0.4 + 0.1 * sup),
                             "similarity": 1.0, "novelty": 0.0, "context": {}, "metadata": {}, "state": "NEW"})))
        _pruned_ev = {"kept": len(_cands), "dropped": 0, "details": []}
        _to_store = [_d for _, _, _d in _cands]
        _kept_seqs = []  # (seq_key, count) survivors; assoc rules prune against these
        if _cands and self.config.get("prune_redundant", True) and not show_all \
                and _closed_keep is not None and _is_subseq is not None:
            try:
                _flags = _closed_keep([(_k, _c) for _k, _c, _ in _cands], mode="closed")
            except Exception:
                _flags = [True] * len(_cands)
            _kept = [t for t, _f in zip(_cands, _flags) if _f]
            _dropped = [t for t, _f in zip(_cands, _flags) if not _f]
            _details = []
            for _k, _c, _ in _dropped:
                _sup = None
                for _kk, _kc, _ in _kept:
                    try:
                        if len(_kk) >= len(_k) and _kc == _c and _is_subseq(_k, _kk):
                            _sup = list(_kk)
                            break
                    except Exception:
                        continue
                _details.append("%s subsumed by %s with equal support %d"
                                % (list(_k), _sup, _c))
            _to_store = [_d for _, _, _d in _kept]
            _kept_seqs = [(_k, _c) for _k, _c, _ in _kept]
            _pruned_ev = {"kept": len(_kept), "dropped": len(_dropped),
                          "details": _details[:20]}
        for _d in _to_store:
            self._store(_d)
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
        try:
            _mp_cfg = max(2, int(self.config.get("max_period", 256)))
        except (TypeError, ValueError):
            _mp_cfg = 256
        if self.config.get("seasonality", True) and estimate_period is not None and len(nums) >= 12:
            try:
                _mp_eff = min(max(2, len(nums) // 2), _mp_cfg)
                try:
                    est = estimate_period(nums, max_period=_mp_eff)
                except TypeError:
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
        if _analyze_seq is not None and len(nums) >= 3:
            try:
                sol = _analyze_seq(nums, steps=1)
                if sol.get("kind") != "unknown" and sol.get("next"):
                    nxt = sol["next"][0]
                    feats = {"kind": sol["kind"], "next": nxt}
                    for k, v in (sol.get("params", {}) or {}).items():
                        if isinstance(v, (int, float, str)):
                            feats[str(k)] = v
                    self._store({"type": "arithmetic", "features": feats,
                                 "sequence": [nxt],
                                 "relationships": {}, "frequency": 1,
                                 "first_seen": None, "last_seen": None, "occurrences": [],
                                 "confidence": float(sol.get("confidence", 0.0)),
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
        if self.config.get("structural", True):
            try:
                _g = _co_graph(labels, window=2) if _co_graph is not None else None
                _comps = _co_comps(_g) if (_co_comps is not None and isinstance(_g, dict)) else []
                try:
                    _nn = len(_g.get("nodes", [])) if isinstance(_g, dict) else 0
                    _ne = len(_g.get("edges", {})) if isinstance(_g, dict) else 0
                    _nc = len(_comps) if isinstance(_comps, list) else 0
                except Exception:
                    _nn, _ne, _nc = 0, 0, 0
                _structural_ev = {"nodes": _nn, "edges": _ne, "components": _nc}
                if _assoc_rules is not None and len(labels) >= 3:
                    try:
                        _txns = [list(labels[_i:_i + 3]) for _i in range(0, len(labels) - 2, 3)]
                        _txns = [_t for _t in _txns if len(_t) == 3]
                    except Exception:
                        _txns = []
                    try:
                        _rules = _assoc_rules(_txns, min_support=min_sup, min_confidence=0.5) if _txns else []
                    except Exception:
                        _rules = []
                    for _rl in (_rules or []):
                        try:
                            _ant, _con = _rl.get("antecedent"), _rl.get("consequent")
                            _sup = float(_rl.get("support", 0.0) or 0.0)
                            _cnf = float(_rl.get("confidence", 0.0) or 0.0)
                            try:
                                _cnt = int(_rl.get("count", min_sup) or min_sup)
                            except (TypeError, ValueError):
                                _cnt = min_sup
                            _rd = {"type": "association",
                                   "features": {"antecedent": _ant, "consequent": _con,
                                                "support": _sup, "confidence": _cnf},
                                   "sequence": [_ant, _con],
                                   "relationships": {}, "frequency": _cnt,
                                   "first_seen": None, "last_seen": None, "occurrences": [],
                                   "confidence": max(0.0, min(1.0, _cnf)),
                                   "similarity": 1.0, "novelty": 0.0,
                                   "context": {}, "metadata": {}, "state": "NEW"}
                            # WS2: an association rule adds nothing when a
                            # kept sequential/recurring pattern with the same
                            # support-count already covers its items.
                            _rdrop = False
                            if self.config.get("prune_redundant", True) and not show_all:
                                try:
                                    _ritems = {str(_ant), str(_con)}
                                    for _kk, _kc in _kept_seqs:
                                        if _kc == _cnt and _ritems <= set(_kk):
                                            _pruned_ev["dropped"] += 1
                                            if len(_pruned_ev["details"]) < 40:
                                                _pruned_ev["details"].append(
                                                    "%s rule subsumed by %s with equal support %d"
                                                    % ([str(_ant), str(_con)], list(_kk), _cnt))
                                            _rdrop = True
                                            break
                                except Exception:
                                    _rdrop = False
                            if not _rdrop:
                                self._store(_rd)
                        except Exception:
                            continue
            except Exception:
                pass
        pats = self.repo.all()
        # _stats() has two shapes: describe_series() -> {"n", "mean", ...}
        # vs the fallback -> {"count", "mean", ...}. Numeric iff we have
        # values AND a real mean (describe yields mean=None when empty).
        try:
            _n_num = int(st.get("count", 0) or st.get("n", 0) or 0)
        except (TypeError, ValueError):
            _n_num = 0
        if _n_num and st.get("mean", None) is not None:
            # Numeric input: a mean is meaningful — cite it with its base.
            try:
                _mean = float(st.get("mean", 0.0))
            except (TypeError, ValueError):
                _mean = 0.0
            expl = ("Discovered %d pattern(s) from %d observation(s) "
                    "(mean %.3f over %d numeric value(s))."
                    % (len(pats), len(rows), _mean, _n_num))
        else:
            # Categorical input: a mean is meaningless (D1) — cite the
            # mode, distinct-value count and entropy instead.
            _cnt = collections.Counter(str(v) for v in values)
            _n = len(values)
            if _cnt and _n:
                _mode, _mode_c = _cnt.most_common(1)[0]
                try:
                    import math as _m
                    _ent = -sum((_c / _n) * _m.log2(_c / _n) for _c in _cnt.values())
                except (ValueError, ZeroDivisionError):
                    _ent = 0.0
                expl = ("Discovered %d pattern(s) from %d observation(s); categorical values: "
                        "mode '%s' (%d of %d), %d distinct value(s), entropy %.3f bits."
                        % (len(pats), len(rows), _mode, _mode_c, _n, len(_cnt), _ent))
            else:
                expl = ("Discovered %d pattern(s) from %d observation(s); no values to summarize."
                        % (len(pats), len(rows)))
        try:
            new_pats = [dict(_p) for _p in pats if str(_p.get("id")) not in ids_before]
        except Exception:
            new_pats = []
        try:
            new_pats.sort(key=lambda _d: str(_d.get("id")))
        except Exception:
            pass
        try:
            try:
                _ncols = len(_numeric_columns(data))
            except Exception:
                _ncols = 0
            skipped = {}
            if len(nums) < 16:
                skipped["regimes"] = "skipped: need >= 16 numeric values, got %d." % len(nums)
            elif not self.config.get("regimes", True):
                skipped["regimes"] = "skipped: disabled via config regimes=False."
            if len(nums) < 12:
                skipped["seasonality"] = "skipped: need >= 12 numeric values, got %d." % len(nums)
            elif not self.config.get("seasonality", True):
                skipped["seasonality"] = "skipped: disabled via config seasonality=False."
            if _ncols < 2:
                skipped["correlation"] = "skipped: need >= 2 numeric columns, got %d." % _ncols
            elif not self.config.get("correlation", True):
                skipped["correlation"] = "skipped: disabled via config correlation=False."
            skipped["multivariate"] = "n/a in discover (see find_anomalies; needs >= 2*mv_window numeric points)."
        except Exception:
            skipped = {"multivariate": "n/a in discover (see find_anomalies)."}
        try:
            ev = {"stats": st, "structural": dict(_structural_ev),
                  "skipped": dict(skipped), "pruned": dict(_pruned_ev)}
        except Exception:
            ev = {"stats": st, "skipped": {"multivariate": "n/a in discover (see find_anomalies)."},
                  "pruned": dict(_pruned_ev)}
        return {"patterns": pats, "count": len(pats), "reason": expl, "explanation": expl,
                "evidence": ev, "new_patterns": new_pats}

    def match(self, observation):
        """Rank all stored patterns by similarity to one observation."""
        row = _rows([observation])[0]
        res = []
        for p in self.repo.all():
            s = _sim(row, p)
            m = s >= _MATCH_THRESHOLD
            feats = _overlap_features(row, p)
            ev = explain_match(row, p.get("id"), s, feats, m, _MATCH_THRESHOLD) if explain_match else "similarity %.3f" % s
            res.append({"pattern_id": p.get("id"), "similarity": s, "matched": m, "evidence": ev, "explanation": ev})
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
        try:
            _vocab = {str(r.get("label", r.get("value", ""))) for r in rows}
        except Exception:
            _vocab = set()
        try:
            _all_pats = self.repo.all()
        except Exception:
            _all_pats = []
        # D2: only patterns sharing label vocabulary with the input are
        # relevant. Unrelated history (e.g. ABC patterns vs a numeric
        # series) must not make a flat series look "novel".
        relevant = []
        for _p in _all_pats:
            if not isinstance(_p, dict):
                continue
            try:
                _pseq = {str(x) for x in (_p.get("sequence", []) or [])}
            except Exception:
                continue
            if _pseq & _vocab:
                relevant.append(_p)
        if _anomaly_detect is not None:
            out = _anomaly_detect(rows, st, zt, relevant)
        else:
            out = [{"index": r["index"], "value": r["value"], "kind": "statistical", "score": 0.9,
                    "causes": ["z-threshold"], "explanation": "outlier"} for r in rows
                   if isinstance(r["value"], (int, float)) and st["stdev"] > 0 and abs((r["value"] - st["mean"]) / st["stdev"]) >= zt]
        try:
            _raws = [r.get("raw") for r in rows if isinstance(r.get("raw"), dict)]
            _ncols = len(_numeric_columns(_raws)) if _raws else 0
        except Exception:
            _ncols = 0
        # D3: the multivariate detector only runs on genuinely
        # multivariate input (>= 2 numeric columns). A univariate series
        # must not go through it.
        if _ncols >= 2 and self.config.get("multivariate", True) and detect_multivariate is not None and _embed_windows is not None:
            try:
                pairs = [(r["index"], r["value"]) for r in rows
                         if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)]
                msize = max(2, int(self.config.get("mv_window", 5)))
                if len(pairs) >= 2 * msize:
                    wvecs, wstarts = _embed_windows([v for _, v in pairs], msize)
                    if wvecs:
                        mv = detect_multivariate(wvecs, threshold=float(self.config.get("mv_threshold", 0.8)))
                        for a in mv.get("anomalies", []):
                            try:
                                _ei = wstarts[a["index"]] + msize - 1
                                _ei = max(0, min(_ei, len(pairs) - 1))
                            except Exception:
                                try:
                                    _ei = wstarts[a["index"]]
                                except Exception:
                                    continue
                            ri = pairs[_ei][0]
                            rv = pairs[_ei][1]
                    dims = mv.get("dims", msize)
                    try:
                        _msev = _severity_of(a["score"]) if _severity_of else "high"
                    except Exception:
                        _msev = "high"
                    out.append({
                        "index": ri, "value": rv, "z": None, "score": a["score"],
                        "kind": "multivariate",
                                "causes": ["window d2=%.2f over %d dims" % (a["d2"], dims)],
                                "explanation": ("Multivariate outlier at window ending index %s: "
                                                "Mahalanobis d2=%.2f across %d dims, score %.2f "
                                                "(threshold %s)." % (ri, a["d2"], dims, a["score"],
                                                                      self.config.get("mv_threshold", 0.8))),
                                "severity": _msev,
                            })
            except Exception:
                pass
        # WS4: robust rolling median/MAD scores (causal) + level shifts.
        # Seasonal phase-centering comes from the strongest stored
        # seasonal pattern, if any qualifies.
        if _robust_detect is not None and self.config.get("robust", True):
            try:
                _per = None
                try:
                    _best = None
                    for _sp in self.repo.all():
                        if not isinstance(_sp, dict) or _sp.get("type") != "seasonal":
                            continue
                        _ff = _sp.get("features", {}) or {}
                        _pp = int(_ff.get("period", 0) or 0)
                        _ss = float(_ff.get("seasonal_strength", 0.0) or 0.0)
                        if _pp >= 2 and _ss >= 0.5 and (_best is None or _ss > _best[0]):
                            _best = (_ss, _pp)
                    if _best is not None:
                        _per = _best[1]
                except Exception:
                    _per = None
                try:
                    _rw = max(1, int(self.config.get("robust_window", 20)))
                except (TypeError, ValueError):
                    _rw = 20
                try:
                    _rt = float(self.config.get("robust_threshold", 3.5))
                except (TypeError, ValueError):
                    _rt = 3.5
                try:
                    out.extend(_robust_detect(rows, window=_rw, threshold=_rt, period=_per) or [])
                except (ValueError, TypeError):
                    pass
            except Exception:
                pass
        if _level_shifts is not None and self.config.get("level_shifts", True):
            try:
                _lw = max(1, int(self.config.get("ls_window", 10)))
            except (TypeError, ValueError):
                _lw = 10
            try:
                _lt = float(self.config.get("ls_threshold", 3.0))
            except (TypeError, ValueError):
                _lt = 3.0
            try:
                out.extend(_level_shifts(rows, window=_lw, threshold_z=_lt) or [])
            except (ValueError, TypeError):
                pass
        # D3: one event produces one record — merge same-index hits.
        out = _merge_anomalies(out)
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
        try:
            import math as _pmath
        except ImportError:
            _pmath = None

        def _keep_lab(_l):
            try:
                if _l is None:
                    return False
                if isinstance(_l, float) and _l != _l:
                    return False
                return True
            except Exception:
                return False

        try:
            mlabels = [_l for _l in labels if _keep_lab(_l)]
        except Exception:
            mlabels = list(labels)
        if build_transition_matrix is not None:
            self._matrix = build_transition_matrix(mlabels)
            cur = current if current is not None else (mlabels[-1] if mlabels else None)
            preds = predict_next(cur, self._matrix, top_k=3) if cur is not None else []
        else:
            cur, preds = current, []
        ctx_preds, log_loss = [], None
        if build_context_model is not None and labels:
            try:
                mo = max(0, int(self.config.get("context_order", 2)))
                clean = [_l for _l in labels if _keep_lab(_l)]
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
        extrap = {}
        if _analyze_seq is not None:
            try:
                vals = [r.get("value") for r in _rows(data)
                        if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)]
                if len(vals) >= 3:
                    extrap = _analyze_seq(vals, steps=3)
            except Exception:
                extrap = {}
        if not isinstance(extrap, dict):
            extrap = {}
        try:
            _tvals = list(vals)
        except Exception:
            _tvals = []
        if self.config.get("trend_forecast", True) and detect_trend is not None:
            try:
                if len(_tvals) >= 3:
                    _tr = detect_trend(_tvals)
                    _slope = float(_tr.get("slope", 0.0) or 0.0)
                    _dir = str(_tr.get("direction", "flat"))
                    _last = float(_tvals[-1])
                    extrap["trend"] = {"source": "trend",
                                       "next": [_last + _slope * _k for _k in (1, 2, 3)],
                                       "slope": _slope, "direction": _dir}
            except Exception:
                pass
        if self.config.get("seasonality", True):
            try:
                _best = None
                for _sp in self.repo.all():
                    if not isinstance(_sp, dict) or _sp.get("type") != "seasonal":
                        continue
                    try:
                        _ff = _sp.get("features", {}) or {}
                        _pp = int(_ff.get("period", 0) or 0)
                        _ss = float(_ff.get("seasonal_strength", 0.0) or 0.0)
                        _sq = list(_sp.get("sequence", []) or [])
                    except (TypeError, ValueError):
                        continue
                    if _pp >= 2 and _ss >= 0.5 and len(_sq) >= _pp:
                        if _best is None or _ss > _best[0]:
                            _best = (_ss, _pp, _sq)
                if _best is not None and _tvals:
                    _ss, _pp, _sq = _best
                    _phase = _sq[len(_tvals) % _pp]
                    if _phase is not None:
                        extrap["seasonal"] = {"source": "seasonal", "next": [_phase], "period": _pp}
            except Exception:
                pass
        return {"current": cur, "predictions": preds, "context": ctx_preds, "log_loss": log_loss,
                "extrapolation": extrap,
                "reason": expl, "explanation": expl, "evidence": {"matrix_states": states}}

    def predict_next(self, data, current=None):
        """Single best next symbol (v2 additive; does not alter predict()).

        Precedence: arithmetic when extrapolation kind != unknown and
        confidence >= 0.9; else context top when its probability >= markov
        top (ties go to context); else markov top; else none. WS6: the
        chosen statistical candidate is held to abstain_threshold /
        min_evidence — below either, the engine abstains rather than
        guessing (source "abstain", next None).
        """
        try:
            _abstain_p = float(self.config.get("abstain_threshold", 0.5))
        except (TypeError, ValueError):
            _abstain_p = 0.5
        try:
            _min_ev = int(self.config.get("min_evidence", 2))
        except (TypeError, ValueError):
            _min_ev = 2
        try:
            _full = self.predict(data, current=current)
        except Exception:
            return {"next": None, "probability": 0.0, "source": "none", "evidence": "no recorded transitions"}
        try:
            _ex = _full.get("extrapolation", {}) or {}
            if isinstance(_ex, dict) and _ex.get("kind") not in (None, "unknown") and _ex.get("next"):
                try:
                    _conf = float(_ex.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    _conf = 0.0
                if _conf >= 0.9:
                    _nxts = list(_ex.get("next") or [])
                    if _nxts:
                        return {"next": _nxts[0], "probability": max(0.0, min(1.0, _conf)),
                                "source": "arithmetic",
                                "evidence": "extrapolation kind %s confidence %.3f." % (_ex.get("kind"), _conf)}
        except Exception:
            pass
        try:
            _ctx = list(_full.get("context", []) or [])
            _mk = list(_full.get("predictions", []) or [])
            _ct = _ctx[0] if _ctx else None
            _mt = _mk[0] if _mk else None
            try:
                _cp = float(_ct.get("probability", 0.0)) if isinstance(_ct, dict) else -1.0
            except (TypeError, ValueError):
                _cp = -1.0
            try:
                _mp = float(_mt.get("probability", 0.0)) if isinstance(_mt, dict) else -1.0
            except (TypeError, ValueError):
                _mp = -1.0
            if isinstance(_ct, dict) and _cp >= 0.0 and (not isinstance(_mt, dict) or _cp >= _mp):
                _pick, _pick_p, _pick_s = _ct, _cp, "context"
            elif isinstance(_mt, dict) and _mp >= 0.0:
                _pick, _pick_p, _pick_s = _mt, _mp, "markov"
            else:
                _pick = None
            if isinstance(_pick, dict):
                try:
                    _tot = _pick.get("total", _pick.get("count", _min_ev))
                    _tot = int(_tot)
                except (TypeError, ValueError):
                    _tot = _min_ev
                _pp = max(0.0, min(1.0, _pick_p))
                if _pp < _abstain_p or _tot < _min_ev:
                    return {"next": None, "probability": _pp, "source": "abstain",
                            "evidence": ("Abstained: top %s candidate '%s' has P=%.3f (N=%d), "
                                         "below abstain_threshold=%.2f / min_evidence=%d."
                                         % (_pick_s, _pick.get("next"), _pp, _tot,
                                            _abstain_p, _min_ev))}
                return {"next": _pick.get("next"), "probability": _pp,
                        "source": _pick_s, "evidence": str(_pick.get("evidence", ""))}
        except Exception:
            pass
        return {"next": None, "probability": 0.0, "source": "none", "evidence": "no recorded transitions"}

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
        """Data-quality report: raw lists, or a file path (parsed, then raw values assessed).

        Never raises on ordinary data. Raises TypeError for non-list /
        non-path input, FileNotFoundError for a missing PathLike.
        """
        if _quality_report is None:
            raise ImportError("nexora.ingestion.quality is required")
        if isinstance(data, os.PathLike) or _is_file_path(data):
            rows = _rows(data)
            return _quality_report([r.get("raw") for r in rows])
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
