from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class VLMClient(ABC):
    @abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[str | Path] = (),
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError


class CallableVLMClient(VLMClient):
    """Adapter for a locally loaded VLM or a project-specific inference function."""

    def __init__(self, fn: Callable[..., dict[str, Any]]) -> None:
        self.fn = fn

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[str | Path] = (),
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        return self.fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[str(p) for p in images],
            **generation_kwargs,
        )

