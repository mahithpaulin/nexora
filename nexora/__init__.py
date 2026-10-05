"""Nexora package: non-neural pattern-recognition engine (v2.0.0)."""
__version__ = "2.0.0"

try:
    from nexora.api.engine import Nexora
except ImportError:  # package-relative fallback
    try:
        from .api.engine import Nexora
    except ImportError:
        Nexora = None

__all__ = ["Nexora", "__version__"]
