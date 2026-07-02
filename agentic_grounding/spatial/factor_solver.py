from __future__ import annotations

import itertools

from agentic_grounding.fusion.object_registry import ObjectRegistry
from agentic_grounding.schemas import CandidateScore, QuerySpec

from .relations import SpatialRelationEngine


class FactorGraphSolver:
    """Small exhaustive solver for query-conditioned target/anchor assignments."""

    def __init__(self, registry: ObjectRegistry, relations: SpatialRelationEngine) -> None:
        self.registry = registry
        self.relations = relations

    def rank(self, query: QuerySpec) -> list[CandidateScore]:
        targets = self.registry.by_category(query.target.category)
        anchor_candidates = [self.registry.by_category(anchor.category) for anchor in query.anchors]
        results: list[CandidateScore] = []
        for target in targets:
            best = CandidateScore(object_id=target.object_id, total=float("-inf"))
            assignments = itertools.product(*anchor_candidates) if anchor_candidates else [()]
            for assignment in assignments:
                anchor_map = {
                    anchor.anchor_id: obj for anchor, obj in zip(query.anchors, assignment)
                }
                relation_scores: list[float] = []
                conflicts: list[str] = []
                evidence: list[dict] = []
                for predicate in query.predicates:
                    if predicate.object not in anchor_map:
                        continue
                    measurement = self.relations.measure(
                        target,
                        predicate.op,
                        anchor_map[predicate.object],
                        frame=predicate.frame,
                        reference_view=predicate.reference_view,
                    )
                    evidence.append(measurement.to_dict())
                    relation_scores.append(measurement.score * predicate.confidence)
                    if predicate.hard and measurement.satisfied is False:
                        conflicts.append(
                            f"{target.object_id} violates {predicate.op}({predicate.object})"
                        )
                semantic = target.category_scores.get(target.category, 0.0)
                relation_score = sum(relation_scores) / max(1, len(relation_scores))
                geometry = target.association_confidence
                total = 0.35 * semantic + 0.50 * relation_score + 0.15 * geometry
                total -= 0.75 * len(conflicts)
                if total > best.total:
                    best = CandidateScore(
                        object_id=target.object_id,
                        total=total,
                        semantic=semantic,
                        relations=relation_score,
                        geometry=geometry,
                        conflicts=conflicts,
                        evidence=evidence,
                    )
            results.append(best)
        return sorted(results, key=lambda item: item.total, reverse=True)

