"""Nexora engine tying memory + anomaly + prediction + explanation.

All sibling discovery/feature/matching imports are guarded
(try/except ImportError -> None); engine falls back to deterministic
stdlib logic when a module is missing.
"""
from __future__ import annotations

import collections
import os
import statistics
from typing import Any

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
    from nexora.discovery.significance import annotate as _sig_annotate
    from nexora.discovery.significance import is_significant as _sig_decide
    try:
        from nexora.discovery.significance import annotation_stats as _sig_batch
    except ImportError:
        _sig_batch = None
except ImportError:
    find_recurring_values = find_frequent_sequences = change_points = None
    _closed_keep = _is_subseq = None
    _sig_annotate = _sig_decide = None
    _sig_batch = None
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
    from nexora.streaming import RunningStats as _RunningStats
    from nexora.streaming import SlidingStats as _SlidingStats
except ImportError:
    _RunningStats = _SlidingStats = None
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
                  "significance": True, "sig_p_threshold": 0.05, "sig_min_lift": 1.5,
                  "sig_shuffles": 199, "sig_seed": 42, "sig_min_n": 30,
                  "stream_capacity": 1024, "change_z": 6.0,
                  "min_data_discover": 1, "min_data_anomalies": 2, "min_data_predict": 2,
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


def _looks_structured_text(s):
    """True when a bare string should be parsed, not kept as one label.

    A plain label like "ABC" stays a single observation; text with a
    comma/newline or starting with "["/"{" is CSV/JSON and goes through
    parse() so discover("A,B,C") == discover(["A","B","C"]).
    """
    t = s.strip()
    return ("," in s) or ("\n" in s) or t.startswith("[") or t.startswith("{")


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
    # v3 (I10): structured text (CSV/JSON) is parsed element-wise;
    # a plain string stays one observation (labels may contain no
    # comma/newline by construction here — match() wraps first anyway).
    if isinstance(data, str) and _parse_path is not None \
            and _looks_structured_text(data):
        try:
            r = _parse_path(data)
            if r:
                return list(r)
        except Exception:
            pass
    # v3: every engine method accepts any iterable of observations
    # (list, tuple, range, generator, ...) element-wise. A single scalar
    # or row-dict is wrapped as one observation; bare strings stay single
    # observations here (file paths handled above, CSV text via parse()).
    if isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        seq = [data]
    elif isinstance(data, (str, bytes)):
        seq = [data]
    elif isinstance(data, tuple):
        seq = list(data)
    else:
        try:
            seq = list(data)
        except TypeError:
            seq = [data]
    if normalize_observations is not None:
        try:
            r = normalize_observations(seq)
            if r:
                return list(r)
            return []
        except Exception:
            pass
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

STATUS_FOUND = "FOUND"
STATUS_NONE = "NONE"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_LOW_CONF = "LOW_CONFIDENCE"

#: Sentinel for "no previous streamed label yet" (WS7).
_STREAM_UNSET = object()

#: Cap on retained online change events (WS7 keeps memory bounded).
_STREAM_CHANGES_CAP = 100


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

    def __init__(self, config: dict | None = None) -> None:
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
        # WS7 incremental streaming state: O(1)/O(window)/O(vocab^2),
        # never the full history (history deque is capacity-capped).
        self._stream_stats = None
        self._stream_window = None
        self._stream_trans = collections.Counter()
        self._stream_totals = collections.Counter()
        self._stream_prev = _STREAM_UNSET
        self._stream_n = 0
        self._stream_changes = []
        self._stream_last_change = -10**12
        try:
            _cap0 = max(1, int((config or {}).get("stream_capacity", 1024)))
        except (TypeError, ValueError, AttributeError):
            _cap0 = 1024
        self._history = collections.deque(maxlen=_cap0)

    def _store(self, p: dict) -> Any:
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

    def discover(self, data: Any, *, show_all: bool = False) -> dict:
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
            _min_d = max(1, int(self.config.get("min_data_discover", 1)))
        except (TypeError, ValueError):
            _min_d = 1
        if len(rows) < _min_d:
            _r = ("INSUFFICIENT_DATA: need >= %d observation(s), got %d; nothing examined."
                  % (_min_d, len(rows)))
            return {"patterns": [], "count": 0, "reason": _r, "explanation": _r,
                    "evidence": {}, "new_patterns": [],
                    "status": STATUS_INSUFFICIENT, "status_reason": _r}
        try:
            ids_before = set(str(_p.get("id")) for _p in self.repo.all() if isinstance(_p, dict))
        except Exception:
            ids_before = set()
        # WS5: count what THIS call stores vs merges (repo.all() mixes
        # history, so status is based on the call's own findings). The
        # closure works on a COPY of ids_before; the original stays
        # pristine for the new_patterns computation below.
        _call_stat = {"stored": 0, "merged": 0}
        _real_store = self._store

        def _store_counted(p, _rs=_real_store, _cst=_call_stat, _seen=set(ids_before)):
            _pid = _rs(p)
            try:
                if str(_pid) in _seen:
                    _cst["merged"] += 1
                else:
                    _cst["stored"] += 1
                    _seen.add(str(_pid))
            except Exception:
                _cst["stored"] += 1
            return _pid
        _structural_ev = {}
        found = []
        if find_recurring_values is not None:
            try:
                found = list(find_recurring_values(values, min_sup) or [])
            except Exception:
                found = []
        if not found:
            cnt = collections.Counter(str(v) for v in values
                                      if v is not None and v == v)
            idxs = {}
            for r in rows:
                if r["value"] is None or r["value"] != r["value"]:
                    continue
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
                # Missing labels never form patterns (None/NaN skipped).
                if c >= min_sup and a is not None and b is not None and a == a and b == b:
                    seqs.append({"sequence": [a, b], "support": c})
        for s in seqs:
            sq = list(s.get("sequence", [])) if isinstance(s, dict) else list(s)
            # Missing labels never form patterns (fixes a [None, None]
            # artifact with a misread support that predates v2).
            if sq and all(_x is None or _x != _x for _x in sq):
                continue
            if isinstance(s, dict):
                feats = s.get("features", {}) or {}
                try:
                    sup = int(feats.get("count", s.get("frequency", s.get("support", min_sup)))
                              or min_sup)
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
        # WS3: significance — annotate every candidate with its null
        # baseline (expected support, lift, permutation p-value) and drop
        # sequential patterns that could easily occur by chance. The
        # per-pattern threshold is Bonferroni-corrected over the
        # candidate set, and shuffles scale up until the p-value
        # resolution reaches it. Filtering needs powered data
        # (n >= sig_min_n); below that we annotate but drop nothing.
        # show_all=True stores everything unvetted. Singletons are never
        # dropped (their shuffle null is degenerate by construction).
        try:
            _sig_thr = float(self.config.get("sig_p_threshold", 0.05))
        except (TypeError, ValueError):
            _sig_thr = 0.05
        try:
            _sig_lift = float(self.config.get("sig_min_lift", 1.5))
        except (TypeError, ValueError):
            _sig_lift = 1.5
        try:
            _sig_nsh = max(1, int(self.config.get("sig_shuffles", 199)))
        except (TypeError, ValueError):
            _sig_nsh = 199
        try:
            _sig_seed = int(self.config.get("sig_seed", 42))
        except (TypeError, ValueError):
            _sig_seed = 42
        try:
            _sig_min_n = max(1, int(self.config.get("sig_min_n", 30)))
        except (TypeError, ValueError):
            _sig_min_n = 30
        _sig_ev = {"annotated": 0, "dropped": 0, "details": [], "skipped": None,
                   "thresholds": {"p": _sig_thr, "lift": _sig_lift,
                                  "shuffles": _sig_nsh, "seed": _sig_seed,
                                  "min_n": _sig_min_n, "alpha_effective": None}}
        if _to_store and not show_all and self.config.get("significance", True) \
                and _sig_annotate is not None and _sig_decide is not None:
            try:
                _multi = sum(1 for _d in _to_store if len(_d.get("sequence", []) or []) >= 2)
                _alpha = _sig_thr / max(1, _multi)
                _sig_ev["thresholds"]["alpha_effective"] = _alpha
                _nsh = _sig_nsh
                while 1.0 / (1.0 + _nsh) > _alpha and _nsh < 999:
                    _nsh = min(999, _nsh * 2 + 1)
                _sig_ev["thresholds"]["shuffles"] = _nsh
                _ann_fn = _sig_batch if _sig_batch is not None else _sig_annotate
                _ann = _ann_fn([dict(_d) for _d in _to_store], labels,
                               n_shuffles=_nsh, seed=_sig_seed, max_patterns=500)
                _by_seq = {}
                for _a in _ann or []:
                    try:
                        _by_seq.setdefault(tuple(str(_x) for _x in (_a.get("sequence", []) or [])), _a)
                    except Exception:
                        continue
                _sigged = []
                for _d in _to_store:
                    try:
                        _key = tuple(str(_x) for _x in (_d.get("sequence", []) or []))
                        _a = _by_seq.get(_key, {})
                        # Method metadata lives OUTSIDE features so drift
                        # snapshots (and dedupe signatures) compare only
                        # the phenomenon, not our statistics about it.
                        _nd = dict(_d)
                        _nd["significance"] = {
                            "expected_support": float(_a.get("expected_support", 0.0)),
                            "lift": float(_a.get("lift", 1.0)),
                            "p_value": float(_a.get("p_value", 1.0)),
                        }
                        _sigged.append(_nd)
                    except Exception:
                        _sigged.append(_d)
                _sig_ev["annotated"] = len(_sigged)
                if len(labels) < _sig_min_n:
                    _sig_ev["skipped"] = ("n=%d below sig_min_n=%d: annotated only, nothing dropped."
                                          % (len(labels), _sig_min_n))
                    _to_store = _sigged
                elif len(_to_store) > 500:
                    _sig_ev["skipped"] = ("too many candidates (%d > 500): annotated only."
                                          % len(_to_store))
                    _to_store = _sigged
                else:
                    _kept2 = []
                    for _d in _sigged:
                        _sq = list(_d.get("sequence", []) or [])
                        if len(_sq) >= 2:
                            try:
                                _sg = _d.get("significance", {}) or {}
                                _ok, _why = _sig_decide(_sg.get("p_value", 1.0),
                                                        _sg.get("lift", 1.0),
                                                        _d.get("frequency", 0),
                                                        min_support=min_sup,
                                                        p_threshold=_alpha,
                                                        min_lift=_sig_lift)
                            except Exception:
                                _ok, _why = True, "significance check errored; kept."
                            if not _ok:
                                _sig_ev["dropped"] += 1
                                if len(_sig_ev["details"]) < 20:
                                    _sig_ev["details"].append("%s dropped: %s" % (_sq, _why))
                                continue
                        _kept2.append(_d)
                    _to_store = _kept2
            except Exception as _se:
                _sig_ev["skipped"] = "significance errored (%s); stored unvetted." % type(_se).__name__
        elif _to_store and show_all:
            _sig_ev["skipped"] = "show_all=True: stored unvetted."
        for _d in _to_store:
            _store_counted(_d)
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
                    _store_counted({"type": "regime",
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
                        _store_counted({"type": "correlation",
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
                    _store_counted({"type": "seasonal",
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
                    _store_counted({"type": "arithmetic", "features": feats,
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
                                _store_counted(_rd)
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
                  "skipped": dict(skipped), "pruned": dict(_pruned_ev),
                  "significance": dict(_sig_ev)}
        except Exception:
            ev = {"stats": st, "skipped": {"multivariate": "n/a in discover (see find_anomalies)."},
                  "pruned": dict(_pruned_ev), "significance": dict(_sig_ev)}
        # WS5: explicit status from this call's own findings.
        _n_found = _call_stat["stored"] + _call_stat["merged"]
        if _n_found > 0:
            _sig_skip = _sig_ev.get("skipped") or ""
            if "sig_min_n" in _sig_skip and not show_all \
                    and self.config.get("significance", True):
                _status = STATUS_LOW_CONF
                _sreason = ("LOW_CONFIDENCE: %d pattern(s) stored but n=%d is below "
                            "sig_min_n=%d, so none is significance-vetted."
                            % (_n_found, len(labels), _sig_min_n))
            else:
                _status = STATUS_FOUND
                _sreason = ("FOUND: %d pattern(s) stored (%d new, %d merged) from %d "
                            "observation(s)." % (_n_found, _call_stat["stored"],
                                                 _call_stat["merged"], len(rows)))
        else:
            _status = STATUS_NONE
            _sreason = ("NONE: nothing stored from %d observation(s): %d sequential "
                        "candidate(s) examined, %d pruned as redundant, %d failed "
                        "significance (min_support=%d)."
                        % (len(rows), len(_cands), _pruned_ev.get("dropped", 0),
                           _sig_ev.get("dropped", 0), min_sup))
        return {"patterns": pats, "count": len(pats), "reason": expl, "explanation": expl,
                "evidence": ev, "new_patterns": new_pats,
                "status": _status, "status_reason": _sreason}

    def match(self, observation: Any) -> list:
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

    def match_top(self, observation: Any) -> dict:
        """Best single match for one observation (I14).

        match() returns the full ranking (kept for compatibility);
        match_top() answers "what is this most like?" with one status
        envelope: {"pattern_id", "similarity", "matched", "explanation",
        "ranked" (total compared), "status", ...}. NONE when nothing is
        stored or nothing reaches the threshold.
        """
        ranked = self.match(observation)
        if not ranked:
            _r = "NONE: no patterns stored; run discover() first."
            return {"pattern_id": None, "similarity": 0.0, "matched": False,
                    "explanation": _r, "ranked": 0, "reason": _r,
                    "status": STATUS_NONE, "status_reason": _r}
        best = ranked[0]
        if best.get("matched"):
            return {"pattern_id": best.get("pattern_id"),
                    "similarity": best.get("similarity", 0.0),
                    "matched": True, "explanation": best.get("explanation", ""),
                    "ranked": len(ranked),
                    "reason": best.get("explanation", ""),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: best of %d (sim=%.3f)." % (
                        len(ranked), best.get("similarity", 0.0))}
        _r = ("NONE: best of %d below threshold %.2f (sim=%.3f)."
              % (len(ranked), _MATCH_THRESHOLD, best.get("similarity", 0.0)))
        return {"pattern_id": best.get("pattern_id"),
                "similarity": best.get("similarity", 0.0),
                "matched": False, "ranked": len(ranked),
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def detect(self, data: Any, include_anomalies: bool = False) -> dict:
        """discover + match in one pass over the data.

        include_anomalies (I21, opt-in): also run find_anomalies and
        attach {"anomalies", "anomaly_count"} — off by default so the
        call stays a single mining pass.
        """
        disc = self.discover(data)
        matches = [m for r in _rows(data) for m in self.match(r)]
        expl = "detect: %d pattern(s), %d match(es) over %d observation(s)." % (disc.get("count", 0), len(matches), len(_rows(data)))
        # WS5: detect reports discover's status (match adds no new claims).
        out = {"patterns": disc.get("patterns", []), "matches": matches, "reason": expl, "explanation": expl, "evidence": disc.get("evidence", {}),
                "status": disc.get("status", STATUS_FOUND),
                "status_reason": "detect: " + str(disc.get("status_reason", disc.get("status", STATUS_FOUND)))}
        if include_anomalies:
            try:
                an = self.find_anomalies(data)
            except Exception:
                an = {"anomalies": [], "count": 0, "status": STATUS_NONE}
            out["anomalies"] = an.get("anomalies", [])
            out["anomaly_count"] = an.get("count", 0)
            out["anomaly_status"] = an.get("status", STATUS_NONE)
        return out

    def find_anomalies(self, data: Any, min_severity: str | None = None) -> dict:
        """Flag statistical outliers + novel/missing sequence transitions.

        min_severity (I13): keep only records at/above this level
        ("low" < "medium" < "high" < "critical"); None keeps all.
        Unknown levels raise ValueError naming the four.
        """
        rows = _rows(data)
        st = _stats([r["value"] for r in rows])
        zt = float(self.config.get("z_threshold", 3.0))
        try:
            _min_a = max(1, int(self.config.get("min_data_anomalies", 2)))
        except (TypeError, ValueError):
            _min_a = 2
        if len(rows) < _min_a:
            _r = ("INSUFFICIENT_DATA: need >= %d observation(s) for anomaly detection, got %d."
                  % (_min_a, len(rows)))
            return {"anomalies": [], "count": 0, "reason": _r, "explanation": _r,
                    "evidence": {"stats": st}, "status": STATUS_INSUFFICIENT,
                    "status_reason": _r}
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
        if min_severity is not None:
            _rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if str(min_severity).lower() not in _rank:
                raise ValueError(
                    "min_severity must be one of low/medium/high/critical, "
                    "got %r" % (min_severity,))
            _cut = _rank[str(min_severity).lower()]
            out = [a for a in out
                   if _rank.get(str(a.get("severity", "low")).lower(), 0) >= _cut]
        expl = "Found %d anomalie(s) (z_threshold=%s)." % (len(out), zt)
        if out:
            _status, _sreason = STATUS_FOUND, ("FOUND: %d anomalie(s) in %d observation(s)."
                                                % (len(out), len(rows)))
        else:
            _status, _sreason = STATUS_NONE, ("NONE: no anomalies in %d observation(s) "
                                               "(z_threshold=%s)." % (len(rows), zt))
        return {"anomalies": out, "count": len(out), "reason": expl, "explanation": expl,
                "evidence": {"stats": st}, "status": _status, "status_reason": _sreason}

    def predict(self, data: Any, current: Any = None) -> dict:
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
        try:
            _min_p = max(1, int(self.config.get("min_data_predict", 2)))
        except (TypeError, ValueError):
            _min_p = 2
        # INSUFFICIENT only when nothing is computable at all: too few
        # labels for transitions AND too few numerics for extrapolation.
        try:
            _nvals_early = sum(1 for _r in _rows(data)
                               if isinstance(_r.get("value"), (int, float))
                               and not isinstance(_r.get("value"), bool))
        except Exception:
            _nvals_early = 0
        if len(mlabels) < _min_p and _nvals_early < 3:
            _r = ("INSUFFICIENT_DATA: need >= %d usable label(s) for prediction, got %d "
                  "(and only %d numeric value(s), need 3 for extrapolation)."
                  % (_min_p, len(mlabels), _nvals_early))
            return {"current": None, "predictions": [], "context": [], "log_loss": None,
                    "extrapolation": {}, "reason": _r, "explanation": _r,
                    "evidence": {"matrix_states": []}, "status": STATUS_INSUFFICIENT,
                    "status_reason": _r}
        if build_transition_matrix is not None:
            self._matrix = build_transition_matrix(mlabels)
            cur = current if current is not None else (mlabels[-1] if mlabels else None)
            preds = predict_next(cur, self._matrix, top_k=3) if cur is not None else []
        else:
            cur, preds = current, []
        ctx_preds, log_loss = [], None
        _ctx_from, _ctx_query = "tail", []
        if build_context_model is not None and labels:
            try:
                mo = max(0, int(self.config.get("context_order", 2)))
                clean = [_l for _l in labels if _keep_lab(_l)]
                self._ctx = build_context_model(clean, mo)
                tail = clean[-mo:] if mo > 0 else []
                # v3: an explicit current conditions the context query.
                # The order-1 field always honored current; the backoff
                # path used to answer from the data tail even when the
                # caller asked about a different (possibly never-seen)
                # symbol. Now the query ends with current — backing off
                # to lower orders when that context was never seen — so
                # both fields answer the same question. Restating the
                # last label changes nothing (tail already ends with it).
                if current is not None and mo > 0 and _keep_lab(current):
                    try:
                        hash(current)
                        _cur_ok = True
                    except TypeError:
                        _cur_ok = False
                    if _cur_ok and (not tail or tail[-1] != current):
                        tail = (list(tail) + [current])[-mo:]
                        _ctx_from = "explicit current"
                _ctx_query = list(tail)
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
        # WS5: explicit status from the top candidates (mirrors the
        # predict_next abstain gates: thin evidence must not read FOUND).
        _best_p, _best_tot = -1.0, 0
        for _cand in (list(preds[:1]) + list(ctx_preds[:1])):
            try:
                if isinstance(_cand, dict):
                    _p = float(_cand.get("probability", -1.0))
                    _t = int(_cand.get("total", _cand.get("count", 0)))
                    if _p > _best_p:
                        _best_p, _best_tot = _p, _t
            except (TypeError, ValueError):
                pass
        try:
            _abst = float(self.config.get("abstain_threshold", 0.5))
        except (TypeError, ValueError):
            _abst = 0.5
        try:
            _min_ev = int(self.config.get("min_evidence", 2))
        except (TypeError, ValueError):
            _min_ev = 2
        _has_extrap = isinstance(extrap, dict) and bool(
            extrap.get("next") or extrap.get("trend")
            or (extrap.get("seasonal") or {}).get("next")
            or (extrap.get("kind") not in (None, "unknown") and extrap.get("next")))
        if not preds and not ctx_preds and _has_extrap:
            _status = STATUS_FOUND
            _sreason = ("FOUND: closed-form extrapolation available "
                        "(no Markov/context transitions from '%s')." % cur)
        elif not preds and not ctx_preds:
            _status = STATUS_NONE
            _sreason = ("NONE: no recorded outgoing transitions from '%s'%s."
                        % (cur, " (arithmetic extrapolation still available)" if _has_extrap else ""))
        elif _best_p < _abst or _best_tot < _min_ev:
            _status = STATUS_LOW_CONF
            _sreason = ("LOW_CONFIDENCE: best P=%.3f (N=%d) below abstain_threshold=%.2f / "
                        "min_evidence=%d after '%s'." % (_best_p, _best_tot, _abst, _min_ev, cur))
        else:
            _status = STATUS_FOUND
            _sreason = ("FOUND: %d Markov + %d context candidate(s) after '%s' (best P=%.3f, N=%d)."
                        % (len(preds), len(ctx_preds), cur, _best_p, _best_tot))
        return {"current": cur, "predictions": preds, "context": ctx_preds, "log_loss": log_loss,
                "extrapolation": extrap,
                "reason": expl, "explanation": expl,
                "evidence": {"matrix_states": states, "context_query": _ctx_query,
                             "context_from": _ctx_from,
                             # I20: perplexity = 2**log_loss (avg branching).
                             "perplexity": (2.0 ** log_loss) if isinstance(log_loss, float) else None},
                "status": _status, "status_reason": _sreason}

    def predict_next(self, data: Any, current: Any = None) -> dict:
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
            return {"next": None, "probability": 0.0, "source": "none", "evidence": "no recorded transitions",
                    "status": STATUS_NONE, "status_reason": "NONE: prediction failed; nothing recorded."}
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
                                "evidence": "extrapolation kind %s confidence %.3f." % (_ex.get("kind"), _conf),
                                "status": STATUS_FOUND,
                                "status_reason": "FOUND: arithmetic extrapolation."}
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
                                            _abstain_p, _min_ev)),
                            "status": STATUS_LOW_CONF,
                            "status_reason": ("LOW_CONFIDENCE: abstained (P=%.3f, N=%d)."
                                              % (_pp, _tot))}
                return {"next": _pick.get("next"), "probability": _pp,
                        "source": _pick_s, "evidence": str(_pick.get("evidence", "")),
                        "status": STATUS_FOUND,
                        "status_reason": "FOUND: %s prediction." % _pick_s}
        except Exception:
            pass
        return {"next": None, "probability": 0.0, "source": "none", "evidence": "no recorded transitions",
                "status": STATUS_NONE, "status_reason": "NONE: no recorded transitions."}

    def get_pattern(self, pid: str) -> dict | None:
        """Return one stored pattern dict (or None)."""
        return self.repo.get(pid)

    def get_history(self, pid: str) -> dict:
        """Occurrences + confidence trail for one pattern."""
        p = self.repo.get(pid)
        if p is None:
            _r = "Pattern not found."
            return {"pattern_id": pid, "occurrences": [], "confidence_trail": [], "reason": _r,
                    "status": STATUS_NONE, "status_reason": "NONE: " + _r}
        occ = p.get("occurrences", []) if isinstance(p, dict) else _get(p, "occurrences", [])
        trail = list(self._trail.get(pid, [p.get("confidence", 0.5) if isinstance(p, dict) else 0.5]))
        return {"pattern_id": pid, "occurrences": occ, "confidence_trail": trail,
                "reason": "History for %s: %d occurrence(s)." % (pid, len(occ)),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: history for %s." % pid}

    def explain(self, result: Any) -> str:
        """Human-readable summary of any result dict."""
        if summarize_result is not None:
            try:
                return summarize_result(result)
            except Exception:
                pass
        return str(result)

    def quality(self, data: Any) -> dict:
        """Data-quality report: raw iterables, or a file path (parsed, then raw values assessed).

        Never raises on ordinary data. Raises TypeError for non-iterable /
        non-path input, FileNotFoundError for a missing PathLike.
        """
        if _quality_report is None:
            raise ImportError("nexora.ingestion.quality is required")
        if isinstance(data, os.PathLike) or _is_file_path(data):
            rows = _rows(data)
            rep = _quality_report([r.get("raw") for r in rows])
        else:
            rep = _quality_report(data)
        if not isinstance(rep, dict):
            return rep
        try:
            _qn = int(rep.get("n", 0) or 0)
        except (TypeError, ValueError):
            _qn = 0
        if _qn == 0:
            rep["status"] = STATUS_INSUFFICIENT
            rep["status_reason"] = "INSUFFICIENT_DATA: no observations to assess."
        else:
            rep["status"] = STATUS_FOUND
            rep["status_reason"] = "FOUND: quality assessed over %d observation(s)." % _qn
        return rep

    def report(self, result: Any, fmt: str = "markdown") -> str:
        """Full human-readable report: fmt="markdown" (default) or "text".

        Unknown formats raise ValueError (I27) instead of silently
        falling back to markdown. Never raises on malformed results
        (coerces with placeholders).
        """
        if fmt not in ("markdown", "text"):
            raise ValueError("fmt must be 'markdown' or 'text', got %r" % (fmt,))
        if fmt == "text":
            if _render_text is None:
                raise ImportError("nexora.explanation.report is required")
            return _render_text(result)
        if _render_markdown is None:
            raise ImportError("nexora.explanation.report is required")
        return _render_markdown(result)

    def save(self, path: Any) -> Any:
        """Persist engine state (config, patterns, trails) to path (atomic write).

        Returns path. Raises ImportError if store module missing, OSError
        on real IO failures.
        """
        if _save_state is None:
            raise ImportError("nexora.memory.store is required")
        return _save_state(path, self.repo, trails=self._trail, config=self.config)

    def load(self, path: Any) -> dict:
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
                "reason": "Loaded %d pattern(s) (schema v%s)." % (self.repo.size(), st.get("version")),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: state loaded."}

    def batch(self, datasets: Any) -> list:
        """Isolated detect() per dataset (fresh engine each); returns outcomes.

        One bad dataset records {"error": ...} and never kills the batch.
        A {name: data} dict is accepted (I26): each outcome's "index" is
        the dataset name. See nexora.api.batch.summarize_batch /
        compare_signatures for rollups and diffs.
        """
        if _batch_process is None:
            raise ImportError("nexora.api.batch is required")
        if isinstance(datasets, dict):
            out = []
            for name, ds in datasets.items():
                try:
                    res = _batch_process([ds], self.config)
                    row = dict(res[0]) if res else {"result": None, "error": "empty"}
                    row["index"] = name
                    out.append(row)
                except Exception as exc:
                    out.append({"index": name, "result": None,
                                "error": "%s: %s" % (type(exc).__name__, exc)})
            return out
        return _batch_process(datasets, self.config)

    def update(self, data: Any, *, detect_changes: bool = True) -> dict:
        """Fold new observations into bounded incremental state (WS7).

        O(1) amortized per observation, O(window + vocab^2 + capacity)
        memory: Welford running stats, a sliding window, bigram
        transition counts, and a capacity-capped recent history. The
        full history is NEVER reprocessed (update() never iterates
        ``self._history``). Online change detection compares the full
        trailing window mean against the long-run mean (|z| >=
        change_z, one event per window cooldown) using past data only.

        Returns {"processed", "total", "stats", "window", "changes"
        (new events this call), "changes_total", "reason"}.
        Stream rows carry GLOBAL indices: history row ["index"] equals
        its ["stream_pos"] (0-based over everything streamed so far),
        so change events and history agree across calls.
        """
        if _RunningStats is None or _SlidingStats is None:
            raise ImportError("nexora.streaming is required")
        import math as _math
        try:
            cap = max(1, int(self.config.get("stream_capacity", 1024)))
        except (TypeError, ValueError):
            cap = 1024
        if self._history.maxlen != cap:
            # Rebuild only when the capacity actually changed.
            self._history = collections.deque(self._history, maxlen=cap)
        if self._stream_stats is None:
            self._stream_stats = _RunningStats()
        try:
            wsize = max(1, int(self.config.get("window", 20)))
        except (TypeError, ValueError):
            wsize = 20
        if self._stream_window is None or self._stream_window.window != wsize:
            self._stream_window = _SlidingStats(wsize)
        try:
            cz = float(self.config.get("change_z", 6.0))
        except (TypeError, ValueError):
            cz = 6.0
        if not cz > 0:
            cz = 6.0
        rows = _rows(data)
        # v3: global stream positions as row indices. _rows() numbers
        # per-call (0..n-1); streaming history must not restart at 0 per
        # chunk, so renumber to base+i == stream_pos before folding.
        try:
            _base = int(self._stream_n)
        except (TypeError, ValueError):
            _base = 0
        for _k, _r in enumerate(rows):
            try:
                _r["index"] = _base + _k
            except Exception:
                pass
        new_changes = []
        for r in rows:
            v = r.get("value")
            lab = r.get("label")
            try:
                lab_ok = lab is not None and lab == lab and hash(lab) is not None
            except TypeError:
                lab_ok = False
            num_ok = (isinstance(v, (int, float)) and not isinstance(v, bool)
                      and _math.isfinite(float(v)))
            if num_ok and detect_changes:
                try:
                    x = float(v)
                    if self._stream_window.n >= wsize and self._stream_stats.n >= 2 * wsize:
                        wm = self._stream_window.mean
                        lm = self._stream_stats.mean
                        ls = self._stream_stats.stdev
                        if wm is not None and lm is not None and ls is not None:
                            if ls == 0.0:
                                fire = wm != lm
                                z = _math.inf if x > lm else (-_math.inf if x < lm else 0.0)
                            else:
                                se = ls / _math.sqrt(wsize)
                                z = (wm - lm) / se if se > 0 else 0.0
                                fire = abs(z) >= cz
                            if fire and (self._stream_n - self._stream_last_change) >= wsize:
                                score = 1.0 if z in (_math.inf, -_math.inf) else min(1.0, abs(z) / (cz * 2.0))
                                ev = {"index": r.get("index"), "value": v, "z": z,
                                      "score": score, "kind": "stream_change",
                                      # index and stream_pos are both global
                                      # (v3: row indices no longer restart).
                                      "stream_pos": self._stream_n,
                                      "causes": ["window mean %.4g vs long-run mean %.4g "
                                                 "(long stdev %.4g, |z|=%.2f >= %.2f)"
                                                 % (wm, lm, ls, z, cz)],
                                      "explanation": ("Online change at index %s (value %s): "
                                                      "trailing-%d mean %.4g deviates |z|=%.2f "
                                                      "from long-run mean %.4g (threshold %.2f)."
                                                      % (r.get("index"), v, wsize, wm, z, lm, cz))}
                                try:
                                    ev["severity"] = _severity_of(score) if _severity_of else "high"
                                except Exception:
                                    ev["severity"] = "high"
                                self._stream_changes.append(ev)
                                if len(self._stream_changes) > _STREAM_CHANGES_CAP:
                                    del self._stream_changes[:-_STREAM_CHANGES_CAP]
                                self._stream_last_change = self._stream_n
                                new_changes.append(ev)
                except Exception:
                    pass
            try:
                self._stream_stats.update(v)
            except Exception:
                pass
            try:
                self._stream_window.update(v)
            except Exception:
                pass
            if lab_ok:
                if self._stream_prev is not _STREAM_UNSET:
                    try:
                        self._stream_trans[(self._stream_prev, lab)] += 1
                        self._stream_totals[self._stream_prev] += 1
                    except TypeError:
                        pass
                self._stream_prev = lab
            try:
                r["stream_pos"] = self._stream_n
            except Exception:
                pass
            self._history.append(r)
            self._stream_n += 1
        try:
            _s = {"n": self._stream_stats.n, "mean": self._stream_stats.mean,
                  "stdev": self._stream_stats.stdev, "missing": self._stream_stats.missing}
        except Exception:
            _s = {"n": 0, "mean": None, "stdev": None, "missing": 0}
        try:
            _w = {"window": self._stream_window.window, "n": self._stream_window.n,
                  "mean": self._stream_window.mean, "stdev": self._stream_window.stdev}
        except Exception:
            _w = {"window": wsize, "n": 0, "mean": None, "stdev": None}
        return {"processed": len(rows), "total": self._stream_n, "stats": _s,
                "window": _w, "changes": new_changes,
                "changes_total": len(self._stream_changes),
                "reason": ("Streamed %d observation(s), %d total; %d new change(s), %d retained."
                           % (len(rows), self._stream_n, len(new_changes),
                              len(self._stream_changes))),
                "status": STATUS_FOUND if rows else STATUS_NONE,
                "status_reason": ("FOUND: streamed %d observation(s)." % len(rows)
                                  if rows else "NONE: empty chunk; state unchanged.")}

    def stream_predict(self, current: Any = None) -> dict:
        """Next-symbol prediction from the incremental stream model (WS7).

        Same MLE math as the batch Markov path, over transition counts
        maintained by update() (no history scan). Returns {"current",
        "predictions", "n", "reason"}; empty predictions when the
        current label has no recorded outgoing transitions.
        """
        if build_transition_matrix is None or predict_next is None:
            raise ImportError("nexora.prediction.markov is required")
        cur = current if current is not None else self._stream_prev
        if cur is _STREAM_UNSET:
            cur = None
        try:
            matrix = {"states": sorted({a for a, _ in self._stream_trans} | {b for _, b in self._stream_trans},
                                       key=str),
                      "counts": dict(self._stream_trans), "probs": {}}
        except Exception:
            matrix = {"states": [], "counts": {}, "probs": {}}
        try:
            preds = predict_next(cur, matrix, top_k=3) if cur is not None else []
        except Exception:
            preds = []
        return {"current": cur, "predictions": preds, "n": self._stream_n,
                "reason": ("Stream model after %d observation(s): %d prediction(s) after '%s'."
                           % (self._stream_n, len(preds), cur)),
                "status": STATUS_FOUND if preds else STATUS_NONE,
                "status_reason": ("FOUND: stream predictions after '%s'." % cur
                                  if preds else "NONE: no stream transitions from '%s'." % cur)}

    # ---- v3 read-out helpers (I5-I9): additive, no behavior change. ----

    def configure(self, config: dict | None = None, **kw) -> dict:
        """Live-update config with fail-fast validation (I11).

        Accepts a dict, kwargs, or both (kwargs win). Unknown keys are
        NOT applied and are named in the result (validate_config
        silently ignores them, which hid typos). Returns {"applied",
        "unknown", "config", "status", ...}.
        """
        if _validate_config is None:
            raise ImportError("nexora.core.config is required")
        merged = dict(config or {})
        merged.update(kw)
        if not merged:
            return {"applied": [], "unknown": [], "config": dict(self.config),
                    "reason": "Config unchanged (nothing given).",
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: config unchanged."}
        unknown = sorted(k for k in merged if k not in DEFAULT_CONFIG)
        try:
            new_cfg = _validate_config(merged)
        except Exception:
            # Validate against current+overlay so partial updates work:
            # unknown keys are dropped before validating.
            known = {k: v for k, v in merged.items() if k in DEFAULT_CONFIG}
            new_cfg = _validate_config(known)
        self.config = new_cfg
        try:
            cap = max(1, int(new_cfg.get("stream_capacity", 1024)))
            if self._history.maxlen != cap:
                self._history = collections.deque(self._history, maxlen=cap)
        except Exception:
            pass
        if not unknown:
            return {"applied": sorted(k for k in merged if k in DEFAULT_CONFIG),
                    "unknown": [], "config": dict(self.config),
                    "reason": "Config updated.",
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: config updated."}
        _r = ("Config updated; unknown key(s) ignored: %s."
              % ", ".join(unknown))
        return {"applied": sorted(k for k in merged if k in DEFAULT_CONFIG),
                "unknown": unknown, "config": dict(self.config),
                "reason": _r, "explanation": _r,
                "status": STATUS_FOUND, "status_reason": "FOUND: " + _r}

    def reset(self) -> dict:
        """Clear memory, trails and stream state (I11). Config is kept."""
        if PatternRepository is None:
            raise ImportError("nexora.memory.repository is required")
        self.repo = PatternRepository()
        self._trail = {}
        self._snaps = {}
        self._matrix = None
        self._ctx = None
        self._stream_stats = None
        self._stream_window = None
        self._stream_trans = collections.Counter()
        self._stream_totals = collections.Counter()
        self._stream_prev = _STREAM_UNSET
        self._stream_n = 0
        self._stream_changes = []
        self._stream_last_change = -10**12
        try:
            cap = max(1, int(self.config.get("stream_capacity", 1024)))
        except (TypeError, ValueError):
            cap = 1024
        self._history = collections.deque(maxlen=cap)
        _r = "Engine reset: memory, trails and stream state cleared."
        return {"reason": _r, "explanation": _r, "status": STATUS_FOUND,
                "status_reason": "FOUND: engine reset."}

    def export_patterns(self) -> dict:
        """JSON-serializable deep copies of stored patterns (I12)."""
        import copy as _copy
        try:
            pats = [ _copy.deepcopy(p) for p in self.repo.all()
                     if isinstance(p, dict)]
        except Exception:
            pats = []
        try:
            import json as _json
            _json.dumps(pats)
        except Exception:
            pats = [p for p in pats if isinstance(p, dict)]
        if pats:
            return {"patterns": pats, "count": len(pats),
                    "reason": "Exported %d pattern(s)." % len(pats),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d pattern(s) exported." % len(pats)}
        _r = "NONE: no patterns stored to export."
        return {"patterns": [], "count": 0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def import_patterns(self, patterns: Any) -> dict:
        """Import pattern dicts (e.g. from export_patterns) (I12).

        Returns {"imported", "skipped", ...}. Malformed entries are
        skipped and counted, never fatal.
        """
        if patterns is None:
            patterns = []
        elif isinstance(patterns, dict):
            patterns = [patterns]
        else:
            try:
                patterns = list(patterns)
            except TypeError:
                _r = "INSUFFICIENT_DATA: nothing importable given."
                return {"imported": 0, "skipped": 0, "reason": _r,
                        "explanation": _r, "status": STATUS_INSUFFICIENT,
                        "status_reason": _r}
        ok, skip = 0, 0
        for p in patterns:
            if not isinstance(p, dict):
                skip += 1
                continue
            try:
                self.repo.import_patterns([dict(p)])
                ok += 1
            except Exception:
                skip += 1
        if ok:
            return {"imported": ok, "skipped": skip,
                    "reason": "Imported %d pattern(s) (%d skipped)." % (ok, skip),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: imported %d pattern(s)." % ok}
        _r = "NONE: imported 0 pattern(s) (%d skipped)." % skip
        return {"imported": 0, "skipped": skip, "reason": _r,
                "explanation": _r, "status": STATUS_NONE,
                "status_reason": _r}

    def describe(self, data: Any) -> dict:
        """Stats + quality snapshot for any iterable (I5).

        Returns {"n", "stats", "quality", "reason", "status", ...}.
        Never raises on ordinary data.
        """
        rows = _rows(data)
        st = _stats([r["value"] for r in rows])
        try:
            q = self.quality([r.get("raw") for r in rows])
        except Exception:
            q = {"quality": 0.0, "reason": "quality unavailable"}
        if not rows:
            _r = "INSUFFICIENT_DATA: no observations to describe."
            return {"n": 0, "stats": st, "quality": q, "reason": _r,
                    "explanation": _r, "status": STATUS_INSUFFICIENT,
                    "status_reason": _r}
        _r = "Described %d observation(s)." % len(rows)
        return {"n": len(rows), "stats": st, "quality": q, "reason": _r,
                "explanation": _r, "status": STATUS_FOUND,
                "status_reason": "FOUND: described %d observation(s)." % len(rows)}

    def top_patterns(self, n: int = 5) -> dict:
        """Top-n stored patterns by confidence (I6)."""
        try:
            all_p = [p for p in self.repo.all() if isinstance(p, dict)]
        except Exception:
            all_p = []
        try:
            n = max(1, int(n))
        except (TypeError, ValueError):
            n = 5
        ranked = sorted(all_p, key=lambda p: (-float(p.get("confidence", 0.0) or 0.0),
                                              str(p.get("id", ""))))[:n]
        if ranked:
            return {"patterns": ranked, "count": len(ranked),
                    "reason": "Top %d of %d stored pattern(s)." % (len(ranked), len(all_p)),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d pattern(s) stored." % len(all_p)}
        _r = "NONE: no patterns stored yet; run discover() first."
        return {"patterns": [], "count": 0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def patterns_by_type(self, ptype: str) -> dict:
        """Stored patterns of one type, e.g. "seasonal" (I6)."""
        try:
            all_p = [p for p in self.repo.all() if isinstance(p, dict)]
        except Exception:
            all_p = []
        hits = [p for p in all_p if str(p.get("type", "")) == str(ptype)]
        if hits:
            return {"patterns": hits, "count": len(hits),
                    "reason": "%d '%s' pattern(s)." % (len(hits), ptype),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d '%s' pattern(s)." % (len(hits), ptype)}
        _r = "NONE: no '%s' patterns stored." % (ptype,)
        return {"patterns": [], "count": 0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def stream_stats(self) -> dict:
        """Current incremental stats without consuming input (I7)."""
        try:
            _s = {"n": self._stream_stats.n, "mean": self._stream_stats.mean,
                  "stdev": self._stream_stats.stdev, "missing": self._stream_stats.missing}
        except Exception:
            _s = {"n": 0, "mean": None, "stdev": None, "missing": 0}
        if self._stream_n:
            return {"stats": _s, "n": self._stream_n,
                    "reason": "Stream state after %d observation(s)." % self._stream_n,
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d streamed." % self._stream_n}
        _r = "NONE: nothing streamed yet; call update() first."
        return {"stats": _s, "n": 0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def stream_changes(self, limit: int = 20) -> dict:
        """Retained online change events, newest last (I7)."""
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 20
        evs = list(self._stream_changes[-limit:])
        if evs:
            return {"changes": evs, "count": len(evs), "total": len(self._stream_changes),
                    "reason": "Showing %d of %d retained change(s)." % (len(evs), len(self._stream_changes)),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d change(s) retained." % len(self._stream_changes)}
        _r = "NONE: no stream changes retained."
        return {"changes": [], "count": 0, "total": 0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def pattern_card(self, pid: str) -> dict:
        """Markdown card for one stored pattern (I8)."""
        p = self.get_pattern(pid)
        if not isinstance(p, dict):
            _r = "NONE: no pattern '%s' stored." % (pid,)
            return {"card": "", "reason": _r, "explanation": _r,
                    "status": STATUS_NONE, "status_reason": _r}
        try:
            card = _pattern_card(p) if _pattern_card is not None else str(p)
        except Exception:
            card = str(p)
        return {"card": card, "pattern": p,
                "reason": "Card for %s." % (pid,),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: card for %s." % (pid,)}

    def solve(self, data: Any, steps: int = 1) -> dict:
        """Closed-form rule for a numeric sequence (I9).

        Wraps discovery.arithmetic: {"kind", "params", "next",
        "confidence", "explanation"} + status (FOUND vs NONE when the
        series fits no rule or is too short).
        """
        if _analyze_seq is None:
            raise ImportError("nexora.discovery.arithmetic is required")
        try:
            steps = max(1, int(steps))
        except (TypeError, ValueError):
            steps = 1
        vals = [r.get("value") for r in _rows(data)
                if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool)]
        try:
            sol = _analyze_seq(vals, steps=steps)
        except Exception:
            sol = {"kind": "unknown", "params": {}, "next": [], "confidence": 0.0,
                   "explanation": "solver errored; no rule."}
        if not isinstance(sol, dict):
            sol = {"kind": "unknown", "params": {}, "next": [], "confidence": 0.0,
                   "explanation": "solver returned nothing usable."}
        if sol.get("kind") not in (None, "unknown") and sol.get("next"):
            return {"kind": sol.get("kind"), "params": sol.get("params", {}),
                    "next": list(sol.get("next") or []),
                    "confidence": float(sol.get("confidence", 0.0) or 0.0),
                    "reason": sol.get("explanation", ""),
                    "explanation": sol.get("explanation", ""),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %s rule." % sol.get("kind")}
        _r = "NONE: no closed-form rule (%s)." % sol.get("explanation", "unknown")
        return {"kind": "unknown", "params": {}, "next": [],
                "confidence": 0.0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def compare(self, other: Any) -> dict:
        """Diff stored patterns vs another engine or pattern list (I22).

        Uses batch.compare_signatures on content fingerprints.
        Returns {"added", "removed", "common", ...} + status.
        """
        if _compare_sigs is None:
            raise ImportError("nexora.api.batch is required")
        try:
            mine = [p for p in self.repo.all() if isinstance(p, dict)]
        except Exception:
            mine = []
        try:
            theirs = [p for p in other.repo.all() if isinstance(p, dict)] \
                if hasattr(other, "repo") else list(other or [])
        except Exception:
            theirs = []
        try:
            diff = _compare_sigs(mine, theirs) or {}
        except Exception:
            diff = {}
        if not isinstance(diff, dict):
            diff = {}
        _r = ("Compared %d own vs %d other pattern(s): %d added, %d removed, %d common."
              % (len(mine), len(theirs), len(diff.get("added", [])),
                 len(diff.get("removed", [])), len(diff.get("common", []))))
        return {"added": list(diff.get("added", [])), "removed": list(diff.get("removed", [])),
                "common": list(diff.get("common", [])),
                "reason": _r, "explanation": _r, "status": STATUS_FOUND,
                "status_reason": "FOUND: " + _r}

    def summarize(self) -> dict:
        """One-dict engine health: memory, stream, config (I23)."""
        try:
            all_p = [p for p in self.repo.all() if isinstance(p, dict)]
        except Exception:
            all_p = []
        by_type = collections.Counter(str(p.get("type", "unknown")) for p in all_p)
        out = {"patterns": len(all_p), "by_type": dict(by_type),
               "stream_n": self._stream_n,
               "stream_changes": len(self._stream_changes),
               "history": len(self._history),
               "config_keys": len(self.config),
               "reason": "%d pattern(s), %d streamed." % (len(all_p), self._stream_n),
               "status": STATUS_FOUND,
               "status_reason": "FOUND: engine summarized."}
        return out

    def forget(self, pid: str) -> dict:
        """Drop one stored pattern by id (I24)."""
        try:
            all_ids = {str(p.get("id")) for p in self.repo.all() if isinstance(p, dict)}
        except Exception:
            all_ids = set()
        if str(pid) not in all_ids:
            _r = "NONE: no pattern '%s' stored." % (pid,)
            return {"forgotten": False, "reason": _r, "explanation": _r,
                    "status": STATUS_NONE, "status_reason": _r}
        try:
            # Repository has no delete; rebuild without the id.
            keep = [p for p in self.repo.all()
                    if isinstance(p, dict) and str(p.get("id")) != str(pid)]
            fresh = PatternRepository()
            fresh.import_patterns(keep)
            self.repo = fresh
            self._trail.pop(str(pid), None)
            self._snaps.pop(str(pid), None)
        except Exception:
            _r = "NONE: could not forget '%s'." % (pid,)
            return {"forgotten": False, "reason": _r, "explanation": _r,
                    "status": STATUS_NONE, "status_reason": _r}
        _r = "Forgot pattern '%s'." % (pid,)
        return {"forgotten": True, "reason": _r, "explanation": _r,
                "status": STATUS_FOUND, "status_reason": "FOUND: " + _r}

    def prune(self, max_total: int = 1000) -> dict:
        """Drop retired patterns while size > max_total (I25).

        Oldest-first among RETIRED states; returns removed ids.
        """
        try:
            max_total = max(0, int(max_total))
        except (TypeError, ValueError):
            raise ValueError("max_total must be an int >= 0")
        try:
            removed = self.repo.prune(max_total=max_total) or []
        except Exception:
            removed = []
        try:
            left = self.repo.size()
        except Exception:
            left = 0
        _r = "Pruned %d pattern(s); %d remain." % (len(removed), left)
        return {"removed": list(removed), "remaining": left,
                "reason": _r, "explanation": _r, "status": STATUS_FOUND,
                "status_reason": "FOUND: " + _r}

    def missing_runs(self, data: Any) -> dict:
        """Stretches of missing values (None/NaN) with start/end/length (I28)."""
        rows = _rows(data)
        runs, start = [], None
        for i, r in enumerate(rows):
            try:
                v = r.get("value")
                missing = v is None or (isinstance(v, float) and v != v)
            except Exception:
                missing = True
            if missing and start is None:
                start = i
            elif not missing and start is not None:
                runs.append({"start": start, "end": i - 1, "length": i - start})
                start = None
        if start is not None:
            runs.append({"start": start, "end": len(rows) - 1,
                         "length": len(rows) - start})
        if runs:
            return {"runs": runs, "count": len(runs),
                    "missing": sum(x["length"] for x in runs), "n": len(rows),
                    "reason": "%d missing run(s), %d value(s) of %d." % (
                        len(runs), sum(x["length"] for x in runs), len(rows)),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d missing run(s)." % len(runs)}
        _r = "NONE: no missing values in %d observation(s)." % len(rows)
        return {"runs": [], "count": 0, "missing": 0, "n": len(rows),
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def runs(self, data: Any) -> dict:
        """Maximal constant-value runs with start/end/count (I29)."""
        rows = _rows(data)
        vals = []
        for r in rows:
            try:
                vals.append(r.get("value"))
            except Exception:
                vals.append(None)
        out, i = [], 0
        while i < len(vals):
            j = i
            try:
                while j + 1 < len(vals) and vals[j + 1] == vals[i] \
                        and not (isinstance(vals[i], float) and vals[i] != vals[i]):
                    j += 1
            except Exception:
                pass
            try:
                key = vals[i]
                disp = None if (isinstance(key, float) and key != key) else key
            except Exception:
                disp = None
            out.append({"value": disp, "start": i, "end": j, "count": j - i + 1})
            i = j + 1
        if out:
            return {"runs": out, "count": len(out), "n": len(rows),
                    "reason": "%d run(s) in %d observation(s)." % (len(out), len(rows)),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d run(s)." % len(out)}
        _r = "NONE: no observations to scan."
        return {"runs": [], "count": 0, "n": 0,
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    # ---- v3 analysis wrappers (batch C): additive read-only views. ----

    def _numeric_values(self, rows):
        """float list of numeric row values (bool/NaN excluded)."""
        out = []
        for r in rows:
            try:
                v = r.get("value")
            except Exception:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if v != v:
                continue
            out.append(float(v))
        return out

    def seasonality(self, data: Any) -> dict:
        """Dominant period + strength for numeric data (I15)."""
        if estimate_period is None or decompose is None:
            raise ImportError("nexora.features.seasonality is required")
        vals = self._numeric_values(_rows(data))
        if len(vals) < 6:
            _r = ("INSUFFICIENT_DATA: need >= 6 numeric values, got %d."
                  % len(vals))
            return {"period": None, "strength": 0.0, "reason": _r,
                    "explanation": _r, "status": STATUS_INSUFFICIENT,
                    "status_reason": _r}
        try:
            mp = min(max(2, len(vals) // 2),
                     max(2, int(self.config.get("max_period", 256))))
        except (TypeError, ValueError):
            mp = min(max(2, len(vals) // 2), 256)
        try:
            est = estimate_period(vals, max_period=mp)
        except TypeError:
            est = estimate_period(vals)
        except Exception:
            est = {"period": None, "strength": 0.0}
        per = (est or {}).get("period")
        if per:
            try:
                dec = decompose(vals, int(per))
            except Exception:
                dec = {}
            return {"period": int(per), "strength": float((est or {}).get("strength", 0.0)),
                    "seasonal_strength": (dec or {}).get("seasonal_strength", 0.0),
                    "trend_strength": (dec or {}).get("trend_strength", 0.0),
                    "reason": "Period %d (strength %.3f)." % (per, (est or {}).get("strength", 0.0)),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: period %d." % per}
        _r = "NONE: no dominant period (strength %.3f)." % float((est or {}).get("strength", 0.0))
        return {"period": None, "strength": float((est or {}).get("strength", 0.0)),
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def correlation(self, data: Any) -> dict:
        """Pairwise numeric-column correlations (I16).

        Needs >= 2 numeric columns (dict rows); otherwise
        INSUFFICIENT_DATA with the column count cited.
        """
        if find_correlation_patterns is None:
            raise ImportError("nexora.features.correlation is required")
        rows = _rows(data)
        raws = [r.get("raw") for r in rows if isinstance(r.get("raw"), dict)]
        cols = _numeric_columns(raws)
        if len(cols) < 2:
            _r = ("INSUFFICIENT_DATA: need >= 2 numeric columns, found %d."
                  % len(cols))
            return {"pairs": [], "count": 0, "columns": sorted(cols),
                    "reason": _r, "explanation": _r,
                    "status": STATUS_INSUFFICIENT, "status_reason": _r}
        try:
            thr = float(self.config.get("corr_threshold", 0.7))
        except (TypeError, ValueError):
            thr = 0.7
        try:
            pats = find_correlation_patterns(cols, threshold=thr) or []
        except Exception:
            pats = []
        pairs = []
        for cp in pats:
            try:
                feats = cp.get("features", {}) or {}
                pairs.append({"a": feats.get("a"), "b": feats.get("b"),
                              "r": feats.get("r"), "strength": feats.get("strength"),
                              "n": feats.get("n")})
            except Exception:
                continue
        if pairs:
            return {"pairs": pairs, "count": len(pairs), "columns": sorted(cols),
                    "reason": "%d correlated pair(s) at |r|>=%.2f." % (len(pairs), thr),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d pair(s)." % len(pairs)}
        _r = "NONE: no pairs at |r|>=%.2f over %d column(s)." % (thr, len(cols))
        return {"pairs": [], "count": 0, "columns": sorted(cols),
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def regimes(self, data: Any) -> dict:
        """Clustered level-regimes of numeric data (I17)."""
        if find_regimes is None:
            raise ImportError("nexora.discovery.clustering is required")
        vals = self._numeric_values(_rows(data))
        if len(vals) < 16:
            _r = ("INSUFFICIENT_DATA: need >= 16 numeric values, got %d."
                  % len(vals))
            return {"regimes": [], "count": 0, "reason": _r,
                    "explanation": _r, "status": STATUS_INSUFFICIENT,
                    "status_reason": _r}
        try:
            size = max(2, int(self.config.get("regime_size", 8)))
            k = max(2, int(self.config.get("n_clusters", 2)))
        except (TypeError, ValueError):
            size, k = 8, 2
        try:
            found = find_regimes(vals, size=size, k=k) or []
        except Exception:
            found = []
        if found:
            return {"regimes": found, "count": len(found),
                    "reason": "%d regime(s), k=%d, window=%d." % (len(found), k, size),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d regime(s)." % len(found)}
        _r = "NONE: no regimes (k=%d, window=%d)." % (k, size)
        return {"regimes": [], "count": 0, "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def change_points(self, data: Any) -> dict:
        """Level/step change points, numeric + label views (I18)."""
        try:
            from nexora.discovery.change_points import change_points as _cp
            from nexora.discovery.change_points import label_change_points as _lcp
        except ImportError:
            _cp = _lcp = None
        if _cp is None:
            raise ImportError("nexora.discovery.change_points is required")
        rows = _rows(data)
        vals = self._numeric_values(rows)
        labels = [r.get("label") for r in rows]
        pts, lab_pts = [], []
        if len(vals) >= 6:
            try:
                _w = max(2, int(self.config.get("ls_window", 10)))
            except (TypeError, ValueError):
                _w = 10
            try:
                pts = _cp(vals, window=_w) or []
            except Exception:
                pts = []
        if _lcp is not None and len(labels) >= 3:
            try:
                lab_pts = _lcp(labels) or []
            except Exception:
                lab_pts = []
        if pts or lab_pts:
            return {"points": pts, "label_points": lab_pts,
                    "count": len(pts) + len(lab_pts),
                    "reason": "%d numeric + %d label change point(s)."
                              % (len(pts), len(lab_pts)),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d change point(s)."
                                     % (len(pts) + len(lab_pts))}
        _r = "NONE: no change points in %d observation(s)." % len(rows)
        return {"points": [], "label_points": [], "count": 0,
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def frequencies(self, data: Any, top_k: int = 10) -> dict:
        """Value-count table for any iterable (I19)."""
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 10
        rows = _rows(data)
        cnt = collections.Counter()
        for r in rows:
            try:
                v = r.get("value")
                key = str(v) if not isinstance(v, str) else v
            except Exception:
                continue
            if v is None or v != v:
                continue
            cnt[key] += 1
        total = sum(cnt.values())
        table = [{"value": k, "count": c,
                  "fraction": (c / total) if total else 0.0}
                 for k, c in cnt.most_common(top_k)]
        if table:
            return {"frequencies": table, "distinct": len(cnt), "n": total,
                    "reason": "%d distinct value(s) in %d observation(s)."
                              % (len(cnt), total),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d distinct value(s)." % len(cnt)}
        _r = "NONE: no countable values."
        return {"frequencies": [], "distinct": 0, "n": 0,
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    def transitions(self, data: Any, top_k: int = 10) -> dict:
        """Bigram transition table with probabilities (I19)."""
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            top_k = 10
        labels = [r.get("label") for r in _rows(data)
                  if r.get("label") is not None and r.get("label") == r.get("label")]
        pairs = collections.Counter(zip(labels, labels[1:]))
        totals = collections.Counter()
        for (a, _b), c in pairs.items():
            totals[a] += c
        table = [{"from": a, "to": b, "count": c,
                  "probability": (c / totals[a]) if totals[a] else 0.0}
                 for (a, b), c in pairs.most_common(top_k)]
        if table:
            return {"transitions": table, "distinct": len(pairs),
                    "reason": "%d distinct transition(s)." % len(pairs),
                    "status": STATUS_FOUND,
                    "status_reason": "FOUND: %d transition(s)." % len(pairs)}
        _r = "NONE: no transitions (need >= 2 labels)."
        return {"transitions": [], "distinct": 0,
                "reason": _r, "explanation": _r,
                "status": STATUS_NONE, "status_reason": _r}

    # ---- v3 numeric views (batch F): additive read-only wrappers. ----

    def histogram(self, data: Any, bins: int = 10) -> dict:
        """Equal-width value histogram (I30)."""
        try:
            bins = max(1, int(bins))
        except (TypeError, ValueError):
            bins = 10
        vals = self._numeric_values(_rows(data))
        if not vals:
            _r = "INSUFFICIENT_DATA: no numeric values for a histogram."
            return {"bins": [], "n": 0, "reason": _r, "explanation": _r,
                    "status": STATUS_INSUFFICIENT, "status_reason": _r}
        lo, hi = min(vals), max(vals)
        width = (hi - lo) / bins if hi > lo else 1.0
        counts = [0] * bins
        for v in vals:
            idx = min(bins - 1, int((v - lo) / width)) if width > 0 else 0
            counts[idx] += 1
        table = [{"low": lo + i * width, "high": lo + (i + 1) * width,
                  "count": c, "fraction": c / len(vals)}
                 for i, c in enumerate(counts)]
        return {"bins": table, "n": len(vals), "min": lo, "max": hi,
                "reason": "%d value(s) in %d bin(s)." % (len(vals), bins),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: histogram over %d value(s)." % len(vals)}

    def zscores(self, data: Any) -> dict:
        """Per-point z vs the batch mean/pstdev (I31)."""
        rows = _rows(data)
        vals = self._numeric_values(rows)
        if len(vals) < 2:
            _r = "INSUFFICIENT_DATA: need >= 2 numeric values for z-scores."
            return {"zscores": [], "n": len(vals), "reason": _r,
                    "explanation": _r, "status": STATUS_INSUFFICIENT,
                    "status_reason": _r}
        import statistics as _st
        mean = _st.fmean(vals)
        sd = _st.pstdev(vals)
        out = []
        for i, r in enumerate(rows):
            try:
                v = r.get("value")
            except Exception:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                continue
            out.append({"index": i, "value": float(v),
                        "z": ((float(v) - mean) / sd) if sd > 0 else 0.0})
        return {"zscores": out, "mean": mean, "stdev": sd, "n": len(out),
                "reason": "%d z-score(s), mean %.4g, stdev %.4g." % (len(out), mean, sd),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: %d z-score(s)." % len(out)}

    def autocorr(self, data: Any, max_lag: int = 10) -> dict:
        """Autocorrelation {lag: r} for lags 1..max_lag (I32)."""
        if autocorrelation is None:
            raise ImportError("nexora.features.temporal is required")
        try:
            max_lag = max(1, int(max_lag))
        except (TypeError, ValueError):
            max_lag = 10
        vals = self._numeric_values(_rows(data))
        if len(vals) < 2:
            _r = "INSUFFICIENT_DATA: need >= 2 numeric values."
            return {"lags": {}, "reason": _r, "explanation": _r,
                    "status": STATUS_INSUFFICIENT, "status_reason": _r}
        try:
            lags = autocorrelation(vals, max_lag=max_lag) or {}
        except Exception:
            lags = {}
        return {"lags": {int(k): float(v) for k, v in lags.items()},
                "reason": "Autocorrelation over %d lag(s)." % len(lags),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: %d lag(s)." % len(lags)}

    def moving_average(self, data: Any, window: int | None = None) -> dict:
        """Trailing moving-average series (I33)."""
        if moving_average is None:
            raise ImportError("nexora.features.temporal is required")
        try:
            window = max(1, int(window if window is not None
                                else self.config.get("window", 20)))
        except (TypeError, ValueError):
            window = 20
        vals = self._numeric_values(_rows(data))
        if not vals:
            _r = "INSUFFICIENT_DATA: no numeric values."
            return {"series": [], "reason": _r, "explanation": _r,
                    "status": STATUS_INSUFFICIENT, "status_reason": _r}
        try:
            series = moving_average(vals, window=window) or []
        except Exception:
            series = []
        return {"series": list(series), "window": window, "n": len(vals),
                "reason": "Moving average (window %d) over %d value(s)."
                          % (window, len(vals)),
                "status": STATUS_FOUND,
                "status_reason": "FOUND: moving average computed."}

    def trend(self, data: Any) -> dict:
        """Least-squares slope/direction/R^2 (I34)."""
        if detect_trend is None:
            raise ImportError("nexora.features.temporal is required")
        vals = self._numeric_values(_rows(data))
        if len(vals) < 2:
            _r = "INSUFFICIENT_DATA: need >= 2 numeric values for a trend."
            return {"slope": 0.0, "direction": "flat", "strength": 0.0,
                    "reason": _r, "explanation": _r,
                    "status": STATUS_INSUFFICIENT, "status_reason": _r}
        try:
            tr = detect_trend(vals) or {}
        except Exception:
            tr = {}
        if not isinstance(tr, dict):
            tr = {}
        _r = "Trend %s (slope %.4g, R^2 %.3f)." % (
            tr.get("direction", "flat"), tr.get("slope", 0.0), tr.get("r2", tr.get("strength", 0.0)))
        return {"slope": float(tr.get("slope", 0.0) or 0.0),
                "direction": str(tr.get("direction", "flat")),
                "strength": float(tr.get("strength", tr.get("r2", 0.0)) or 0.0),
                "reason": _r, "explanation": _r,
                "status": STATUS_FOUND, "status_reason": "FOUND: " + _r}
