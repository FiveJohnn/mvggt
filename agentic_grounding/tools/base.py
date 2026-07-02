from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def schema(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}

