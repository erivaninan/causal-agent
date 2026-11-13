"""Donut regression-discontinuity estimation."""

from .estimator import DonutRDD, estimate_effect

__all__ = ["DonutRDD", "estimate_effect"]
