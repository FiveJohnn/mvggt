from .base import ConceptSegmenter, Detector, Segmenter

__all__ = [
    "ConceptSegmenter",
    "Detector",
    "HuggingFaceGroundingDINO",
    "HTTPConceptSegmenter",
    "SAM3ConceptSegmenter",
    "SAMBoxSegmenter",
    "Segmenter",
]


def __getattr__(name: str):
    """Keep optional model libraries out of lightweight imports."""
    if name == "HuggingFaceGroundingDINO":
        from .grounding_dino import HuggingFaceGroundingDINO

        return HuggingFaceGroundingDINO
    if name == "HTTPConceptSegmenter":
        from .http_segmenter import HTTPConceptSegmenter

        return HTTPConceptSegmenter
    if name == "SAM3ConceptSegmenter":
        from .sam3 import SAM3ConceptSegmenter

        return SAM3ConceptSegmenter
    if name == "SAMBoxSegmenter":
        from .sam_box import SAMBoxSegmenter

        return SAMBoxSegmenter
    raise AttributeError(name)
