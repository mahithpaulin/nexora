"""Transparent pattern scoring for Nexora (no black boxes).

Formula (documented):
  frequency_score   = frequency / total_n           (0..1, 0 if total_n <= 0)
  similarity_score  = clamp(similarity)             (0..1, None -> 0.0)
  stability_score   = clamp(consistency)            (0..1, None -> 0.0)
  novelty_score     = 1 - frequency_score           (0..1, rare = novel)
  predictive_score  = clamp(predictive_strength)    (0..1, None -> 0.0)
  recency_score     = clamp(recency)                (0..1, None -> 0.0)
  noise_penalty     = clamp(noise)                  (0..1, None -> 0.0)
  uncertainty_penalty = clamp(uncertainty)          (0..1, None -> 0.0)
  positive = weighted mean of the six scores above
             (weights normalized to sum 1)
  confidence = clamp01(positive - w_noise*noise_penalty
                       - w_unc*uncertainty_penalty)  (0..1)

Default weights: frequency 0.25, similarity 0.15, stability 0.20,
novelty 0.10, predictive 0.15, recency 0.15 (sum 1.0),
noise 0.5, uncertainty 0.5 (penalty scales). Custom weights dict
merges over these keys and the six positive weights are renormalized.
Every component is returned separately for inspection.
"""

import math
from typing import Any, Dict, Optional

DEFAULT_WEIGHTS = {
    "frequency": 0.25,
    "similarity": 0.15,
    "stability": 0.20,
    "novelty": 0.10,
    "predictive": 0.15,
    "recency": 0.15,
    "noise": 0.5,
    "uncertainty": 0.5,
}

_POS_KEYS = ("frequency", "similarity", "stability",
             "novelty", "predictive", "recency")


def _c01(x: Any, default: float = 0.0) -> float:
    """Clamp a value to 0..1; None/NaN/garbage yields default."""
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def score_pattern(frequency: Any, total_n: Any, consistency: Any,
                  similarity: Any, recency: Any, predictive_strength: Any,
                  noise: Any, uncertainty: Any,
                  weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Score one pattern transparently from evidence and quality signals.

    What it computes: component scores each in 0..1 (higher = stronger,
    except noise_penalty/uncertainty_penalty where higher = worse) plus
    overall confidence in 0..1 (higher = more trustworthy) and a plain
    English explanation string. Missing/None inputs are treated as 0.0
    (no evidence) and never raise.
    """
    try:
        f = max(0.0, float(frequency or 0))
    except (TypeError, ValueError):
        f = 0.0
    try:
        n = max(0.0, float(total_n or 0))
    except (TypeError, ValueError):
        n = 0.0
    frequency_score = _c01(f / n) if n > 0 else 0.0
    similarity_score = _c01(similarity)
    stability_score = _c01(consistency)
    novelty_score = _c01(1.0 - frequency_score)
    predictive_score = _c01(predictive_strength)
    recency_score = _c01(recency)
    noise_penalty = _c01(noise)
    uncertainty_penalty = _c01(uncertainty)

    w = dict(DEFAULT_WEIGHTS)
    if isinstance(weights, dict):
        for k, v in weights.items():
            try:
                if v is not None:
                    w[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    pos_w = {k: max(0.0, w.get(k, 0.0)) for k in _POS_KEYS}
    s = sum(pos_w.values())
    if s > 0:
        pos_w = {k: v / s for k, v in pos_w.items()}
    else:
        pos_w = {k: 1.0 / len(_POS_KEYS) for k in _POS_KEYS}
    positive = (pos_w["frequency"] * frequency_score
                + pos_w["similarity"] * similarity_score
                + pos_w["stability"] * stability_score
                + pos_w["novelty"] * novelty_score
                + pos_w["predictive"] * predictive_score
                + pos_w["recency"] * recency_score)
    w_noise = max(0.0, w.get("noise", 0.5))
    w_unc = max(0.0, w.get("uncertainty", 0.5))
    confidence = _c01(positive - w_noise * noise_penalty
                      - w_unc * uncertainty_penalty)
    explanation = (
        f"confidence={confidence:.3f}: weighted mean {positive:.3f} of "
        f"freq={frequency_score:.3f}, sim={similarity_score:.3f}, "
        f"stab={stability_score:.3f}, nov={novelty_score:.3f}, "
        f"pred={predictive_score:.3f}, rec={recency_score:.3f} "
        f"minus noise {noise_penalty:.3f}x{w_noise:.2f} and "
        f"uncertainty {uncertainty_penalty:.3f}x{w_unc:.2f}."
    )
    return {
        "confidence": confidence,
        "frequency_score": frequency_score,
        "similarity_score": similarity_score,
        "stability_score": stability_score,
        "novelty_score": novelty_score,
        "predictive_score": predictive_score,
        "recency_score": recency_score,
        "noise_penalty": noise_penalty,
        "uncertainty_penalty": uncertainty_penalty,
        "explanation": explanation,
    }
