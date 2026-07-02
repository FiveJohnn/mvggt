"""Agentic zero-shot multi-view 3D grounding toolkit.

The package is intentionally independent from the supervised MVGGT referring
head.  Foundation models are accessed through small, replaceable adapters.
"""

from .schemas import Object3D, ObjectObservation, QuerySpec

__all__ = ["Object3D", "ObjectObservation", "QuerySpec"]

