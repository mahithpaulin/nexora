"""Nexora human-readable reporting (Phase 7). Self-contained, stdlib-free."""

MAX_PATTERNS_MD = 20
MAX_ANOMALIES = 20
MAX_PREDICTIONS = 10
MAX_MATCHES = 5
MAX_LIST_ITEMS = 10
MAX_FEATURE_ITEMS = 6


def _get(obj, key, default=None):
    try:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def _s(v, default="?"):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            return v
        return str(v)
    except Exception:
        return default


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _escape_cell(v):
    try:
        t = _s(v)
    except Exception:
        t = "?"
    return t.replace("|", "/").replace("\n", " ").replace("\r", "")


def _plain(v):
    try:
        t = _s(v)
    except Exception:
        t = "?"
    return (
        t.replace("#", "").replace("*", "").replace("`", "'").replace("|", "/").replace("\n", " ").replace("\r", "")
    )


def _float_or(default, v):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _capped(items, limit=10):
    shown = items[:limit]
    rem = len(items) - len(shown)
    return shown, (rem if rem > 0 else 0)


def pattern_card(pattern):
    """Markdown card for one pattern. Never raises."""
    try:
        pid = _s(_get(pattern, "id", "?"))
        ptype = _s(_get(pattern, "type", "?"))
        freq = _get(pattern, "frequency", _get(pattern, "freq", _get(pattern, "count", "?")))
        freq_s = _s(freq)
        conf = _get(pattern, "confidence", "?")
        conf_s = _s(conf)
        state = _s(_get(pattern, "state", "?"))
        support = _get(pattern, "support", _get(pattern, "support_count", _get(pattern, "count", None)))
        if support is None:
            support = freq if freq_s != "?" else "?"
        support_s = _s(support)

        feats = _get(pattern, "features", {})
        if not isinstance(feats, dict):
            feats = {}
        try:
            feat_items = sorted(feats.items(), key=lambda kv: str(kv[0]))
        except Exception:
            feat_items = list(feats.items())
        feat_total = len(feat_items)
        feat_shown = feat_items[:MAX_FEATURE_ITEMS]
        feat_rem = feat_total - len(feat_shown)

        occ_raw = _as_list(_get(pattern, "occurrences", []))
        occ_count = len(occ_raw)
        occ_idx = []
        for o in occ_raw:
            try:
                if isinstance(o, dict) and "index" in o:
                    occ_idx.append(_s(o.get("index")))
                else:
                    occ_idx.append(_s(o))
            except Exception:
                occ_idx.append("?")
        occ_shown, occ_rem = _capped(occ_idx, MAX_LIST_ITEMS)

        fol_raw = _as_list(_get(pattern, "followed_by", _get(pattern, "followedBy", [])))
        pre_raw = _as_list(_get(pattern, "preceded_by", _get(pattern, "precededBy", [])))
        fol_s = [_s(x) for x in fol_raw]
        pre_s = [_s(x) for x in pre_raw]
        fol_shown, fol_rem = _capped(fol_s, MAX_LIST_ITEMS)
        pre_shown, pre_rem = _capped(pre_s, MAX_LIST_ITEMS)

        if feat_shown:
            feats_s = ", ".join("{}={}".format(_s(k), _s(v)) for k, v in feat_shown)
            if feat_rem > 0:
                feats_s += " +{} more".format(feat_rem)
        else:
            feats_s = "none" if feat_total == 0 else "?"
        occ_s = ", ".join(occ_shown)
        if occ_rem > 0:
            occ_s = (occ_s + " " if occ_s else "") + "+{} more".format(occ_rem)
        fol_txt = ", ".join(fol_shown) if fol_shown else "none"
        if fol_rem > 0:
            fol_txt += " +{} more".format(fol_rem)
        pre_txt = ", ".join(pre_shown) if pre_shown else "none"
        if pre_rem > 0:
            pre_txt += " +{} more".format(prem)

        if feat_shown:
            topk, topv = feat_shown[0][0], feat_shown[0][1]
            why = "observed {} time(s) with frequency {} and confidence {}; top feature {}={}.".format(
                occ_count, freq_s, conf_s, _s(topk), _s(topv)
            )
        else:
            why = "observed {} time(s) with frequency {} and confidence {}.".format(occ_count, freq_s, conf_s)

        lines = [
            "## {} `{}`".format(pid, ptype),
            "- frequency: {}".format(freq_s),
            "- confidence: {}".format(conf_s),
            "- state: {}".format(state),
            "- support: {}".format(support_s),
            "- features ({}): {}".format(feat_total, feats_s),
            "- occurrences: {} [{}]".format(occ_count, occ_s),
            "- followed_by: {}".format(fol_txt),
            "- preceded_by: {}".format(pre_txt),
            "- why: {}".format(why),
        ]
        return "\n".join(lines)
    except Exception:
        try:
            return "## {} `{}`\n- why: unavailable.".format(_s(_get(pattern, "id", "?")), _s(_get(pattern, "type", "?")))
        except Exception:
            return "## ? `?`\n- why: unavailable."


def _sorted_patterns(patterns):
    try:
        return sorted(list(patterns), key=lambda p: _s(_get(p, "id", "?")))
    except Exception:
        try:
            return list(patterns)
        except Exception:
            return []


def _sorted_anomalies(anomalies):
    def key(a):
        try:
            idx = _get(a, "index", "?")
            if idx is None:
                return (1, "?")
            try:
                return (0, int(idx))
            except Exception:
                try:
                    return (0, float(idx))
                except Exception:
                    return (1, str(idx))
        except Exception:
            return (1, "?")

    try:
        return sorted(list(anomalies), key=key)
    except Exception:
        return _as_list(anomalies)


def _sorted_matches(matches):
    def sim(m):
        v = _get(m, "similarity", _get(m, "score", _get(m, "confidence", 0)))
        return _float_or(0.0, v)

    try:
        return sorted(list(matches), key=sim, reverse=True)
    except Exception:
        return _as_list(matches)


def _collections(result):
    try:
        patterns = _as_list(_get(result, "patterns", []))
    except Exception:
        patterns = []
    try:
        matches = _as_list(_get(result, "matches", []))
    except Exception:
        matches = []
    try:
        anomalies = _as_list(_get(result, "anomalies", []))
    except Exception:
        anomalies = []
    try:
        predictions = _as_list(_get(result, "predictions", []))
    except Exception:
        predictions = []
    overall = _get(result, "explanation", _get(result, "overall_explanation", _get(result, "overall", _get(result, "summary", ""))))
    if overall is None:
        overall = ""
    try:
        overall = str(overall)
    except Exception:
        overall = ""
    return patterns, matches, anomalies, predictions, overall


def render_markdown(result):
    """Full markdown report (caps: 20 patterns / 20 anomalies / 10 predictions / 5 matches). Never raises."""
    try:
        patterns, matches, anomalies, predictions, overall = _collections(result)
        np_, nm_, na_, npr_ = len(patterns), len(matches), len(anomalies), len(predictions)
        out = ["# Nexora report", ""]
        summary = "Found {} pattern(s), {} match(es), {} anomalie(s), {} prediction(s).".format(np_, nm_, na_, npr_)
        if overall.strip():
            summary += " Overall: {}".format(_s(overall).replace("\n", " "))
        out.append(summary)
        out.append("")

        out.append("## Patterns ({})".format(np_))
        out.append("")
        sp = _sorted_patterns(patterns)
        for p in sp[:MAX_PATTERNS_MD]:
            try:
                out.append(pattern_card(p))
            except Exception:
                out.append("## ? `?`")
            out.append("")
        if np_ > MAX_PATTERNS_MD:
            out.append("+{} more pattern(s)".format(np_ - MAX_PATTERNS_MD))
            out.append("")

        out.append("## Anomalies ({})".format(na_))
        out.append("")
        if na_ == 0:
            out.append("None.")
            out.append("")
        else:
            out.append("| index | value | kind | score | why |")
            out.append("| --- | --- | --- | --- | --- |")
            for a in _sorted_anomalies(anomalies)[:MAX_ANOMALIES]:
                idx = _escape_cell(_get(a, "index", "?"))
                val = _escape_cell(_get(a, "value", "?"))
                kind = _escape_cell(_get(a, "kind", "?"))
                score = _escape_cell(_get(a, "score", _get(a, "anomaly_score", "?")))
                why = _escape_cell(_get(a, "why", _get(a, "explanation", _get(a, "reason", "?"))))
                out.append("| {} | {} | {} | {} | {} |".format(idx, val, kind, score, why))
            if na_ > MAX_ANOMALIES:
                out.append("")
                out.append("+{} more anomalie(s)".format(na_ - MAX_ANOMALIES))
            out.append("")

        out.append("## Predictions ({})".format(npr_))
        out.append("")
        if npr_ == 0:
            out.append("None.")
            out.append("")
        else:
            for pr in predictions[:MAX_PREDICTIONS]:
                nxt = _escape_cell(_get(pr, "next", _get(pr, "value", _get(pr, "predicted", "?"))))
                prob = _escape_cell(_get(pr, "prob", _get(pr, "probability", _get(pr, "confidence", "?"))))
                ev = _get(pr, "evidence", _get(pr, "support", _get(pr, "based_on", "")))
                ev_list = _as_list(ev) if isinstance(ev, (list, tuple)) else ([str(ev)] if ev not in ("", None) else [])
                ev_shown, ev_rem = _capped([_s(x) for x in ev_list], MAX_LIST_ITEMS)
                ev_s = ", ".join(_escape_cell(x) for x in ev_shown)
                if ev_rem > 0:
                    ev_s += " +{} more".format(ev_rem)
                out.append("- next: {} | prob: {} | evidence: {}".format(nxt, prob, ev_s if ev_s else "none"))
            if npr_ > MAX_PREDICTIONS:
                out.append("+{} more prediction(s)".format(npr_ - MAX_PREDICTIONS))
            out.append("")

        out.append("## Matches ({})".format(nm_))
        out.append("")
        if nm_ == 0:
            out.append("None.")
            out.append("")
        else:
            for m in _sorted_matches(matches)[:MAX_MATCHES]:
                mid = _escape_cell(_get(m, "pattern_id", _get(m, "id", _get(m, "pattern", "?"))))
                sim = _escape_cell(_get(m, "similarity", _get(m, "score", _get(m, "confidence", "?"))))
                idx = _get(m, "index", None)
                extra_s = " | index: {}".format(_escape_cell(idx)) if idx is not None else ""
                out.append("- {}: similarity {}{}".format(mid, sim, extra_s))
            if nm_ > MAX_MATCHES:
                out.append("+{} more match(es)".format(nm_ - MAX_MATCHES))
            out.append("")
        return "\n".join(out).rstrip() + "\n"
    except Exception:
        try:
            return "# Nexora report\n\nUnavailable (malformed result).\n"
        except Exception:
            return "# Nexora report\n"


def render_text(result):
    """Plain-text report (no markdown chars). Same caps. Never raises."""
    try:
        patterns, matches, anomalies, predictions, overall = _collections(result)
        np_, nm_, na_, npr_ = len(patterns), len(matches), len(anomalies), len(predictions)
        out = ["Nexora report", ""]
        summary = "Summary: {} patterns, {} matches, {} anomalies, {} predictions.".format(np_, nm_, na_, npr_)
        if _plain(overall).strip():
            summary += " Overall: {}".format(_plain(overall).replace("\n", " "))
        out.append(summary)
        out.append("")

        out.append("Patterns ({}):".format(np_))
        sp = _sorted_patterns(patterns)
        if np_ == 0:
            out.append("  none")
        for p in sp[:MAX_PATTERNS_MD]:
            try:
                pid = _plain(_get(p, "id", "?"))
                ptype = _plain(_get(p, "type", "?"))
                freq = _plain(_get(p, "frequency", _get(p, "freq", _get(p, "count", "?"))))
                conf = _plain(_get(p, "confidence", "?"))
                state = _plain(_get(p, "state", "?"))
                feats = _get(p, "features", {})
                if not isinstance(feats, dict):
                    feats = {}
                try:
                    fitems = sorted(feats.items(), key=lambda kv: str(kv[0]))[:MAX_FEATURE_ITEMS]
                except Exception:
                    fitems = []
                feats_s = ", ".join("{}={}".format(_plain(k), _plain(v)) for k, v in fitems) if fitems else "none"
                if len(feats) > MAX_FEATURE_ITEMS:
                    feats_s += " +{} more".format(len(feats) - MAX_FEATURE_ITEMS)
                occ = _as_list(_get(p, "occurrences", []))
                occ_idx = []
                for o in occ:
                    if isinstance(o, dict) and "index" in o:
                        occ_idx.append(_plain(o.get("index")))
                    else:
                        occ_idx.append(_plain(o))
                oshown, orem = _capped(occ_idx, MAX_LIST_ITEMS)
                occ_s = ", ".join(oshown)
                if orem > 0:
                    occ_s = (occ_s + " " if occ_s else "") + "+{} more".format(orem)
                fol = [_plain(x) for x in _as_list(_get(p, "followed_by", _get(p, "followedBy", [])))]
                pre = [_plain(x) for x in _as_list(_get(p, "preceded_by", _get(p, "precededBy", [])))]
                fshown, frem = _capped(fol, MAX_LIST_ITEMS)
                pshown, prem = _capped(pre, MAX_LIST_ITEMS)
                fol_s = ", ".join(fshown) if fshown else "none"
                if frem > 0:
                    fol_s += " +{} more".format(frem)
                pre_s = ", ".join(pshown) if pshown else "none"
                if prem > 0:
                    pre_s += " +{} more".format(prem)
                out.append("  pattern {} type {} freq {} conf {} state {}".format(pid, ptype, freq, conf, state))
                out.append("    features ({}): {}".format(len(feats), feats_s))
                out.append("    occurrences: {} [{}]".format(len(occ), occ_s))
                out.append("    followed by: {}".format(fol_s))
                out.append("    preceded by: {}".format(pre_s))
                out.append("    why: observed {} times with frequency {} and confidence {}.".format(len(occ), freq, conf))
            except Exception:
                out.append("  pattern unavailable")
        if np_ > MAX_PATTERNS_MD:
            out.append("  +{} more patterns".format(np_ - MAX_PATTERNS_MD))
        out.append("")

        out.append("Anomalies ({}):".format(na_))
        if na_ == 0:
            out.append("  none")
        for a in _sorted_anomalies(anomalies)[:MAX_ANOMALIES]:
            idx = _plain(_get(a, "index", "?"))
            val = _plain(_get(a, "value", "?"))
            kind = _plain(_get(a, "kind", "?"))
            score = _plain(_get(a, "score", _get(a, "anomaly_score", "?")))
            why = _plain(_get(a, "why", _get(a, "explanation", _get(a, "reason", "?"))))
            out.append("  index {} value {} kind {} score {} why {}".format(idx, val, kind, score, why))
        if na_ > MAX_ANOMALIES:
            out.append("  +{} more anomalies".format(na_ - MAX_ANOMALIES))
        out.append("")

        out.append("Predictions ({}):".format(npr_))
        if npr_ == 0:
            out.append("  none")
        for pr in predictions[:MAX_PREDICTIONS]:
            nxt = _plain(_get(pr, "next", _get(pr, "value", _get(pr, "predicted", "?"))))
            prob = _plain(_get(pr, "prob", _get(pr, "probability", _get(pr, "confidence", "?"))))
            ev = _get(pr, "evidence", _get(pr, "support", _get(pr, "based_on", "")))
            ev_list = _as_list(ev) if isinstance(ev, (list, tuple)) else ([str(ev)] if ev not in ("", None) else [])
            eshown, erem = _capped([_plain(x) for x in ev_list], MAX_LIST_ITEMS)
            ev_s = ", ".join(eshown) if eshown else "none"
            if erem > 0:
                ev_s += " +{} more".format(erem)
            out.append("  next {} prob {} evidence {}".format(nxt, prob, ev_s))
        if npr_ > MAX_PREDICTIONS:
            out.append("  +{} more predictions".format(npr_ - MAX_PREDICTIONS))
        out.append("")

        out.append("Matches ({}):".format(nm_))
        if nm_ == 0:
            out.append("  none")
        for m in _sorted_matches(matches)[:MAX_MATCHES]:
            mid = _plain(_get(m, "pattern_id", _get(m, "id", _get(m, "pattern", "?"))))
            sim = _plain(_get(m, "similarity", _get(m, "score", _get(m, "confidence", "?"))))
            out.append("  pattern {} similarity {}".format(mid, sim))
        if nm_ > MAX_MATCHES:
            out.append("  +{} more matches".format(nm_ - MAX_MATCHES))
        out.append("")
        return "\n".join(out).rstrip() + "\n"
    except Exception:
        try:
            return "Nexora report\n\nUnavailable (malformed result).\n"
        except Exception:
            return "Nexora report\n"
