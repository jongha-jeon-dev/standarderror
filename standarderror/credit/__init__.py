"""Credit structures: pricing a junior claim on a single asset."""

from .tranche import (attachment_point, expected_shortfall_rate, required_fall,
                      simulate_shortfall_rate)

__all__ = ["attachment_point", "required_fall", "expected_shortfall_rate",
           "simulate_shortfall_rate"]
