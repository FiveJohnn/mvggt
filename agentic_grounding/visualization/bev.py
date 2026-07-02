from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentic_grounding.fusion.object_registry import ObjectRegistry


def render_bev(
    registry: ObjectRegistry,
    output_path: str | Path | None = None,
    size: int = 1024,
    padding: int = 60,
    target_ids: set[str] | None = None,
    anchor_ids: set[str] | None = None,
    distractor_ids: set[str] | None = None,
) -> Image.Image:
    """Render an x-y bird's-eye summary.

    Callers should rotate points into a gravity-aligned frame before using this
    renderer when the reconstruction coordinate system is arbitrary.
    """
    objects = list(registry)
    canvas = Image.new("RGB", (size, size), "white")
    if not objects:
        return canvas
    all_min = np.min(np.stack([obj.aabb_min[:2] for obj in objects]), axis=0)
    all_max = np.max(np.stack([obj.aabb_max[:2] for obj in objects]), axis=0)
    span = np.maximum(all_max - all_min, 1e-5)

    def project(point: np.ndarray) -> tuple[float, float]:
        normalized = (point[:2] - all_min) / span
        x = padding + normalized[0] * (size - 2 * padding)
        y = size - padding - normalized[1] * (size - 2 * padding)
        return float(x), float(y)

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    target_ids = target_ids or set()
    anchor_ids = anchor_ids or set()
    distractor_ids = distractor_ids or set()
    for obj in objects:
        if obj.object_id in target_ids:
            color = (220, 45, 45)
        elif obj.object_id in anchor_ids:
            color = (30, 100, 220)
        elif obj.object_id in distractor_ids:
            color = (230, 170, 20)
        else:
            color = (90, 90, 90)
        x0, y1 = project(obj.aabb_min)
        x1, y0 = project(obj.aabb_max)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"{obj.object_id}:{obj.category}"
        center = project(obj.centroid)
        draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=color)
        draw.text((center[0] + 6, center[1] - 6), label, fill=color, font=font)
    draw.line((padding, size - padding, padding + 50, size - padding), fill="black", width=2)
    draw.text((padding, size - padding + 8), "BEV x-y (not a metric ruler)", fill="black", font=font)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
    return canvas

