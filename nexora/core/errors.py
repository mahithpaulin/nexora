"""Nexora error hierarchy."""
class NexoraError(Exception):
    """Base class for all Nexora errors."""
class EmptyDataError(NexoraError, ValueError):
    """Raised when input data is empty."""
class InsufficientDataError(NexoraError, ValueError):
    """Raised when data has fewer items than required."""
    def __init__(self, message="", needed=None, got=None):
        super().__init__(message)
        self.needed = needed
        self.got = got
class InvalidConfigError(NexoraError, ValueError):
    """Raised when a config value is invalid."""
    def __init__(self, message="", key=None):
        super().__init__(message)
        self.key = key
class SingularDataError(NexoraError, ValueError):
    """Raised when data is singular or degenerate."""
