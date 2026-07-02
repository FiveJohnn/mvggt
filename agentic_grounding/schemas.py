from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


RelationFrame = Literal["world", "gravity", "view_dependent", "object_centric", "unknown"]


@dataclass
class TargetSpec:
    category: str
    attributes: list[str] = field(default_factory=list)


@dataclass
class AnchorSpec:
    anchor_id: str
    category: str
    attributes: list[str] = field(default_factory=list)


@dataclass
class PredicateSpec:
    op: str
    subject: str = "target"
    object: str = ""
    frame: RelationFrame = "unknown"
    confidence: float = 1.0
    hard: bool = True
    reference_view: int | None = None


@dataclass
class QuerySpec:
    query_id: str
    raw_text: str
    target: TargetSpec
    anchors: list[AnchorSpec] = field(default_factory=list)
    predicates: list[PredicateSpec] = field(default_factory=list)
    hard_distractor_policy: dict[str, bool] = field(
        default_factory=lambda: {"same_category": True, "similar_attributes": True}
    )
    compiler_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuerySpec":
        if not isinstance(data.get("target"), dict):
            raise ValueError("Query compiler output must contain a target object")
        target = TargetSpec(**data["target"])
        anchors = [AnchorSpec(**item) for item in data.get("anchors", [])]
        predicates = [PredicateSpec(**item) for item in data.get("predicates", [])]
        return cls(
            query_id=str(data.get("query_id", "")),
            raw_text=str(data.get("raw_text", "")),
            target=target,
            anchors=anchors,
            predicates=predicates,
            hard_distractor_policy=data.get(
                "hard_distractor_policy",
                {"same_category": True, "similar_attributes": True},
            ),
            compiler_metadata=data.get("compiler_metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys([self.target.category, *[a.category for a in self.anchors]]))


@dataclass
class Detection2D:
    view_id: int
    detection_id: str
    category: str
    score: float
    box_xyxy: np.ndarray
    phrase: str = ""


@dataclass
class ObjectObservation:
    observation_id: str
    view_id: int
    category: str
    category_score: float
    box_xyxy: np.ndarray
    mask: np.ndarray
    appearance_feature: np.ndarray | None = None
    points_world: np.ndarray | None = None
    geometry_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Object3D:
    object_id: str
    category_scores: dict[str, float]
    observation_ids: list[str]
    visible_views: list[int]
    points_world: np.ndarray
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    appearance_feature: np.ndarray | None = None
    association_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return max(self.category_scores, key=self.category_scores.get)

    @property
    def size(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size))


@dataclass
class RelationMeasurement:
    subject_id: str
    predicate: str
    object_id: str
    frame: str
    satisfied: bool | None
    score: float
    measurements: dict[str, float | str | bool | None]
    confidence: float = 1.0
    supporting_views: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScore:
    object_id: str
    total: float
    semantic: float = 0.0
    attributes: float = 0.0
    relations: float = 0.0
    geometry: float = 0.0
    conflicts: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

