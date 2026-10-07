"""Result container for a Nexora analysis run."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

ResultStatus = Literal["FOUND", "NONE", "INSUFFICIENT_DATA", "LOW_CONFIDENCE"]


class Pattern(TypedDict, total=False):
    """Stored pattern shape (all keys optional for backward compatibility)."""

    id: str
    type: str
    features: Dict[str, Any]
    sequence: List[Any]
    relationships: Dict[str, Any]
    frequency: int
    first_seen: Optional[int]
    last_seen: Optional[int]
    occurrences: List[int]
    confidence: float
    similarity: float
    novelty: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    state: str
    significance: Dict[str, float]
    score_detail: Dict[str, Any]


class AnomalyRecord(TypedDict, total=False):
    """One anomaly record (merged kinds joined with '+')."""

    index: int
    value: Any
    z: Optional[float]
    score: float
    kind: str
    causes: List[str]
    explanation: str
    severity: str


class Prediction(TypedDict, total=False):
    """One ranked next-symbol candidate (calibrated, WS6)."""

    next: Any
    probability: float
    evidence: str
    order: int
    count: int
    total: int
    ci95: List[float]


class MatchEntry(TypedDict, total=False):
    """One ranked match of an observation against a stored pattern."""

    pattern_id: Optional[str]
    similarity: float
    matched: bool
    evidence: str
    explanation: str


class StatusEnvelope(TypedDict, total=False):
    """Status fields present on every v2 dict-result (WS5)."""

    status: str
    status_reason: str


def _clamp01(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        v = float(x)
        if v != v:
            return 0.0
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0


def _ser(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "to_dict"):
        try:
            return x.to_dict()
        except Exception:
            pass
    if isinstance(x, dict):
        return {str(k): _ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ser(v) for v in x]
    return x


@dataclass
class Result:
    """Outcome of one detection pass.

    Fields: patterns (detected Pattern/dict list), matches (index ->
    pattern assignments), anomalies (flagged rows with reason/score),
    predictions (forecast dicts), confidence (overall run confidence
    0..1, 1 = fully trusted), explanation (human-readable summary).
    """

    patterns: List[Any] = field(default_factory=list)
    matches: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        for attr in ("patterns", "matches", "anomalies", "predictions"):
            if getattr(self, attr) is None:
                setattr(self, attr, [])
        if self.explanation is None:
            self.explanation = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this Result to a plain JSON-compatible dict."""
        return {
            "patterns": _ser(self.patterns),
            "matches": _ser(self.matches),
            "anomalies": _ser(self.anomalies),
            "predictions": _ser(self.predictions),
            "confidence": self.confidence,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Result":
        """Rebuild a Result from a to_dict() dict; missing keys use defaults."""
        if d is None:
            raise ValueError("from_dict() received None")
        if not isinstance(d, dict):
            raise TypeError("from_dict() expects a dict")
        return cls(
            patterns=d.get("patterns") or [],
            matches=d.get("matches") or [],
            anomalies=d.get("anomalies") or [],
            predictions=d.get("predictions") or [],
            confidence=_clamp01(d.get("confidence", 0.0)),
            explanation=str(d.get("explanation") or ""),
        )
