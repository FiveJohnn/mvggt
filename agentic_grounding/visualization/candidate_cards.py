from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_candidate_cards(
    image_paths: Sequence[str | Path],
    candidates: Sequence[dict],
    output_path: str | Path | None = None,
    tile_size: tuple[int, int] = (256, 224),
) -> Image.Image:
    width, height = tile_size
    cards: list[Image.Image] = []
    font = ImageFont.load_default()
    for candidate in candidates:
        view_id = int(candidate["view_id"])
        box = [int(round(v)) for v in candidate["box_xyxy"]]
        image = Image.open(image_paths[view_id]).convert("RGB")
        box[0] = max(0, box[0]); box[1] = max(0, box[1])
        box[2] = min(image.width, box[2]); box[3] = min(image.height, box[3])
        crop = image.crop(tuple(box)).resize((width, height - 28), Image.Resampling.BILINEAR)
        card = Image.new("RGB", (width, height), "white")
        card.paste(crop, (0, 28))
        draw = ImageDraw.Draw(card)
        draw.text((6, 8), str(candidate["object_id"]), fill="black", font=font)
        cards.append(card)
    if not cards:
        return Image.new("RGB", tile_size, "white")
    canvas = Image.new("RGB", (width * len(cards), height), "white")
    for index, card in enumerate(cards):
        canvas.paste(card, (index * width, 0))
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
    return canvas

