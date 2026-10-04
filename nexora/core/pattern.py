"""Pattern record for the Nexora engine.

Stores one detected pattern with its evidence (sequence, occurrences),
quality scores (confidence 0..1, similarity 0..1|None, novelty 0..1),
and lifecycle state (NEW/OBSERVED/CONFIRMED/ESTABLISHED/EVOLVING/STALE/RETIRED).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

LIFECYCLE_STATES = (
    "NEW", "OBSERVED", "CONFIRMED", "ESTABLISHED",
    "EVOLVING", "STALE", "RETIRED",
)


def make_pattern_id(counter: int) -> str:
    """Build a sequential pattern id like "P-001".

    What it computes: zero-padded sequential identifier from a 1-based
    counter in creation order. No score involved.
    """
    n = int(counter)
    if n < 0:
        n = 0
    return f"P-{n:03d}"


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:  # NaN
            return default
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


@dataclass
class Pattern:
    """Single detected pattern with evidence, scores, and lifecycle state."""

    id: str = ""
    type: str = "statistical"
    features: Dict[str, Any] = field(default_factory=dict)
    sequence: list = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 0
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    occurrences: List[int] = field(default_factory=list)
    confidence: float = 0.0
    similarity: Optional[float] = None
    novelty: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: str = "NEW"

    def __post_init__(self) -> None:
        try:
            self.frequency = max(0, int(self.frequency or 0))
        except (TypeError, ValueError):
            self.frequency = 0
        self.confidence = _clamp01(self.confidence, 0.0)
        self.novelty = _clamp01(self.novelty, 0.0)
        if self.similarity is None:
            pass
        else:
            try:
                self.similarity = _clamp01(self.similarity, 0.0)
            except (TypeError, ValueError):
                self.similarity = None
        for attr in ("features", "relationships", "context", "metadata"):
            if getattr(self, attr) is None:
                setattr(self, attr, {})
        for attr in ("sequence", "occurrences"):
            if getattr(self, attr) is None:
                setattr(self, attr, [])
        if self.state is None:
            self.state = "NEW"
        else:
            self.state = str(self.state).upper() or "NEW"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this Pattern to a plain JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pattern":
        """Rebuild a Pattern from a to_dict() dict; missing keys use defaults."""
        if d is None:
            raise ValueError("from_dict() received None")
        if not isinstance(d, dict):
            raise TypeError("from_dict() expects a dict")
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)
