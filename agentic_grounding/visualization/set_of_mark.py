from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_set_of_mark(
    image: str | Path | Image.Image,
    marks: Iterable[dict],
    output_path: str | Path | None = None,
) -> Image.Image:
    canvas = (image.copy() if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for mark in marks:
        box = tuple(float(value) for value in mark["box_xyxy"])
        label = str(mark["object_id"])
        color = tuple(mark.get("color", (255, 64, 64)))
        draw.rectangle(box, outline=color, width=3)
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill="white", font=font)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
    return canvas

