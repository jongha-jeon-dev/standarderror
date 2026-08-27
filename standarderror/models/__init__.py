"""Models: reservoir computing, its interpretable cousin, and the baselines.

    from standarderror.models import ESN, ESNConfig, NGRC, NGRCConfig, baselines, metrics
"""

from . import baselines, metrics, tune
from .esn import ESN, ESNConfig
from .ngrc import NGRC, NGRCConfig

__all__ = ["ESN", "ESNConfig", "NGRC", "NGRCConfig", "baselines", "metrics",
           "tune"]
