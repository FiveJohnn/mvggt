from .base import GeometryBackend, GeometryResult, load_geometry, save_geometry

__all__ = [
    "GeometryBackend",
    "GeometryResult",
    "MVGGTBackend",
    "VGGTBackend",
    "VGGTOmegaBackend",
    "load_geometry",
    "save_geometry",
]


def __getattr__(name: str):
    """Import torch-heavy model adapters only when they are requested."""
    if name == "MVGGTBackend":
        from .mvggt_backend import MVGGTBackend

        return MVGGTBackend
    if name == "VGGTBackend":
        from .vggt_backend import VGGTBackend

        return VGGTBackend
    if name == "VGGTOmegaBackend":
        from .vggt_omega_backend import VGGTOmegaBackend

        return VGGTOmegaBackend
    raise AttributeError(name)
