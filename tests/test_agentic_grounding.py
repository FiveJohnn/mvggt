import unittest

import numpy as np

from agentic_grounding.fusion.associate import AssociationConfig, associate_observations
from agentic_grounding.fusion.object_registry import ObjectRegistry
from agentic_grounding.geometry.base import GeometryResult
from agentic_grounding.geometry.common import invert_extrinsics
from agentic_grounding.query.compiler import QueryCompiler
from agentic_grounding.schemas import Object3D, ObjectObservation
from agentic_grounding.spatial.relations import SpatialRelationEngine
from agentic_grounding.vlm.base import CallableVLMClient


def observation(observation_id, view_id, center):
    rng = np.random.default_rng(view_id + len(observation_id))
    points = rng.normal(center, 0.01, size=(100, 3)).astype(np.float32)
    return ObjectObservation(
        observation_id=observation_id,
        view_id=view_id,
        category="chair",
        category_score=0.9,
        box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        mask=np.ones((10, 10), dtype=bool),
        points_world=points,
        geometry_confidence=0.9,
    )


class AssociationTests(unittest.TestCase):
    def test_cross_view_merge_and_same_view_cannot_link(self):
        observations = [
            observation("a", 0, [0, 0, 0]),
            observation("b", 1, [0, 0, 0]),
            observation("c", 0, [1, 0, 0]),
        ]
        clusters = associate_observations(
            observations,
            AssociationConfig(voxel_size=0.05, min_pair_score=0.2),
        )
        memberships = [set(item.observation_id for item in cluster) for cluster in clusters]
        self.assertIn({"a", "b"}, memberships)
        self.assertIn({"c"}, memberships)


class RelationTests(unittest.TestCase):
    def test_view_dependent_left(self):
        points_a = np.array([[-1.0, 0.0, 2.0], [-0.9, 0.1, 2.0]], dtype=np.float32)
        points_b = np.array([[1.0, 0.0, 2.0], [0.9, 0.1, 2.0]], dtype=np.float32)
        a = Object3D("O001", {"chair": 1.0}, ["a"], [0], points_a, points_a.mean(0), points_a.min(0), points_a.max(0))
        b = Object3D("O002", {"table": 1.0}, ["b"], [0], points_b, points_b.mean(0), points_b.min(0), points_b.max(0))
        geometry = GeometryResult(
            points_world=np.zeros((1, 2, 2, 3), dtype=np.float32),
            depth=np.ones((1, 2, 2), dtype=np.float32),
            intrinsics=np.array([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]]),
            camera_to_world=np.eye(4, dtype=np.float32)[None],
            confidence=np.ones((1, 2, 2), dtype=np.float32),
        )
        result = SpatialRelationEngine(geometry).measure(
            a, "left_of", b, frame="view_dependent", reference_view=0
        )
        self.assertTrue(result.satisfied)


class GeometryTests(unittest.TestCase):
    def test_invert_3x4_extrinsics(self):
        world_to_camera = np.array(
            [[[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=np.float32,
        )
        camera_to_world = invert_extrinsics(world_to_camera)
        self.assertEqual(camera_to_world.shape, (1, 4, 4))
        np.testing.assert_allclose(camera_to_world[0, :3, 3], [2.0, 0.0, 0.0])


class QueryCompilerTests(unittest.TestCase):
    def test_compile_query_graph(self):
        def fake_vlm(**_):
            return {
                "target": {"category": "chair", "attributes": ["brown"]},
                "anchors": [{"anchor_id": "a0", "category": "table", "attributes": []}],
                "predicates": [{
                    "op": "left_of",
                    "subject": "target",
                    "object": "a0",
                    "frame": "view_dependent",
                    "confidence": 0.9,
                    "hard": True,
                    "reference_view": 0,
                }],
            }

        compiler = QueryCompiler(CallableVLMClient(fake_vlm))
        query = compiler.compile("the brown chair left of the table", "q1")
        self.assertEqual(query.categories, ["chair", "table"])
        self.assertEqual(query.predicates[0].frame, "view_dependent")


if __name__ == "__main__":
    unittest.main()
