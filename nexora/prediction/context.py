"""Variable-order Markov context model with backoff (stdlib only)."""
import math

try:
    from nexora.prediction.calibration import calibrate as _calibrate
except ImportError:  # pragma: no cover - package always ships it
    _calibrate = None


def _is_missing_label(x):
    """True for None or float NaN (treated as missing, skipped)."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def build_context_model(labels: list, max_order: int = 3):
    """Count next-label frequencies for all context lengths 0..max_order.

    Order 0 uses key () for unconditional frequencies. N-grams touching
    a missing label (None/NaN) or an unhashable label are skipped (never
    crashes). Raises ValueError on negative max_order or non-sequences.
    """
    if not isinstance(max_order, int) or isinstance(max_order, bool) \
            or max_order < 0:
        raise ValueError("max_order must be a non-negative int")
    if not isinstance(labels, (list, tuple)):
        raise ValueError("labels must be a list/tuple")
    seq = list(labels)
    orders: dict = {o: {} for o in range(max_order + 1)}
    for i in range(len(seq)):
        nxt = seq[i]
        if _is_missing_label(nxt):
            continue
        try:
            hash(nxt)
        except TypeError:
            continue
        for o in range(max_order + 1):
            if i < o:
                continue
            ctx = tuple(seq[i - o:i])
            if any(_is_missing_label(c) for c in ctx):
                continue
            try:
                bucket = orders[o].setdefault(ctx, {})
                bucket[nxt] = bucket.get(nxt, 0) + 1
            except TypeError:
                continue
    total = sum(sum(d.values()) for d in orders.get(0, {}).values())
    return {"orders": orders, "max_order": max_order,
            "reason": f"counted orders 0..{max_order} over {len(seq)} "
                      f"labels ({total} order-0 events)"}


def predict_with_context(model: dict, context: list, top_k: int = 3):
    """Backoff prediction: longest usable suffix of context.

    Uses the longest suffix of `context` present in the model (orders
    min(len(context), max_order) down to 0); P = count/total at that
    order. Returns [{"next", "probability" (0..1), "order" (context
    length used), "evidence"}] sorted by -probability (ties by repr),
    at most top_k entries. Empty list when nothing matches.
    """
    if top_k is None or not isinstance(top_k, int) \
            or isinstance(top_k, bool) or top_k <= 0:
        return []
    orders = model.get("orders", {}) if isinstance(model, dict) else {}
    max_order = model.get("max_order", 0) if isinstance(model, dict) else 0
    ctx = list(context) if isinstance(context, (list, tuple)) else []
    start = min(len(ctx), max_order if isinstance(max_order, int) else 0)
    for o in range(start, -1, -1):
        suffix = tuple(ctx[len(ctx) - o:]) if o > 0 else ()
        try:
            dist = orders.get(o, {}).get(suffix)
        except (TypeError, AttributeError):
            continue
        if not dist:
            continue
        total = sum(dist.values())
        if total <= 0:
            continue
        ranked = sorted(dist.items(),
                        key=lambda kv: (-kv[1], repr(kv[0])))
        out = []
        for nxt, c in ranked[:top_k]:
            entry = {"next": nxt, "probability": c / total, "order": o,
                     "evidence": f"order {o} context {suffix!r} seen "
                                 f"{total}x: {c}/{total}"}
            # WS6 calibration: MLE probability unchanged, plus uncertainty.
            if _calibrate is not None:
                try:
                    cal = _calibrate(c, total)
                    entry["count"] = cal["count"]
                    entry["total"] = cal["total"]
                    entry["ci95"] = cal["ci95"]
                except Exception:
                    entry["count"], entry["total"] = c, total
            else:
                entry["count"], entry["total"] = c, total
            out.append(entry)
        return out
    return []


def sequence_log_loss(model: dict, labels: list):
    """Mean base-2 negative log-prob of each label under backoff.

    Returns float|None (None when nothing predictable). Skips missing
    actuals, empty backoff, and zero-probability (unseen) tokens.
    Lower = better (0 = perfect). Deterministic, never crashes.
    """
    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        return None
    if not isinstance(model, dict):
        return None
    orders = model.get("orders", {})
    max_order = model.get("max_order", 0)
    if not isinstance(max_order, int) or isinstance(max_order, bool):
        max_order = 0
    seq = list(labels)
    loss, count = 0.0, 0
    for i in range(len(seq)):
        actual = seq[i]
        if _is_missing_label(actual):
            continue
        try:
            hash(actual)
        except TypeError:
            continue
        ctx = seq[max(0, i - max_order):i] if max_order > 0 else []
        start = min(len(ctx), max_order)
        dist = None
        for o in range(start, -1, -1):
            suffix = tuple(ctx[len(ctx) - o:]) if o > 0 else ()
            try:
                cand = orders.get(o, {}).get(suffix)
            except (TypeError, AttributeError):
                continue
            if cand and sum(cand.values()) > 0:
                dist = cand
                break
        if not dist:
            continue
        total = sum(dist.values())
        try:
            p = dist.get(actual, 0) / total if total > 0 else 0.0
        except TypeError:
            continue
        if p <= 0.0:
            continue
        loss += -math.log2(p)
        count += 1
    if count == 0:
        return None
    return loss / count
