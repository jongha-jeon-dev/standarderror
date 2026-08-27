"""Linear algebra as it behaves in floating point, rather than as it is defined.

Every function here exists because some calculation a working data scientist
would write returns the wrong answer, and the piece of theory that explains why
is worth more than the definition it came from.
"""

from . import conditioning

__all__ = ["conditioning"]
