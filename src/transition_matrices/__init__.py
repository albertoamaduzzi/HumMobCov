"""
src/transition_matrices
=======================
Geohash-grid transition matrix and presence matrix computation pipeline.

Main entry point
----------------
>>> from src.transition_matrices import TransitionPipeline
>>> pipeline = TransitionPipeline(dataset, geohash_precision=5)
>>> pipeline.run_period("15 jan - 15 march")
"""

from .transition_pipeline import (  # noqa: F401
    TransitionPipeline,
    build_time_bins,
    compute_presence_matrix,
    compute_transition_matrix,
)
