from __future__ import annotations

import base64
import json
import mimetypes
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentic_grounding.io_utils import extract_json_object

from .base import VLMClient


def _image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleVLMClient(VLMClient):
    """Small dependency-free client for OpenAI-compatible chat endpoints.

    It also works with local servers such as vLLM or SGLang when they expose
    the standard ``/v1/chat/completions`` endpoint.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[str | Path] = (),
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in images
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": generation_kwargs.pop("temperature", 0.0),
            "response_format": {"type": "json_object"},
            **generation_kwargs,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        message = result["choices"][0]["message"]["content"]
        if isinstance(message, dict):
            return message
        return extract_json_object(message)

