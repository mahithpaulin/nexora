"""Nexora package: non-neural pattern-recognition engine (v3.1.0)."""
__version__ = "3.1.0"

try:
    from nexora.api.engine import Nexora
except ImportError:  # package-relative fallback
    try:
        from .api.engine import Nexora
    except ImportError:
        Nexora = None

__all__ = ["Nexora", "__version__"]
