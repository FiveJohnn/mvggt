from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from agentic_grounding.perception.sam3 import SAM3ConceptSegmenter


def _encode_mask(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_handler(segmenter: SAM3ConceptSegmenter):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/segment":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            image = Image.open(io.BytesIO(base64.b64decode(request["image_base64"]))).convert("RGB")
            masks, boxes, scores = segmenter.segment_concept(image, request["prompt"])
            response = {
                "masks_png_base64": [_encode_mask(mask) for mask in masks],
                "boxes_xyxy": boxes.tolist(),
                "scores": scores.tolist(),
            }
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A003
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    segmenter = SAM3ConceptSegmenter()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(segmenter))
    print(f"SAM 3 service listening at http://{args.host}:{args.port}/segment", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

