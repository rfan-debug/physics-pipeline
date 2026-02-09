import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Mock genesis before importing task_generator
# We need to ensure that when task_generator imports genesis, it gets our mock
if "genesis" not in sys.modules:
    sys.modules["genesis"] = MagicMock()

import genesis as gs

# Define mock morphs
gs.morphs = MagicMock()
gs.morphs.Box = MagicMock(return_value="mock_box_morph")
gs.morphs.Sphere = MagicMock(return_value="mock_sphere_morph")
gs.morphs.Cylinder = MagicMock(return_value="mock_cylinder_morph")

from vla_synthesis.src.task_generator import TaskGenerator

class TestTaskGenerator(unittest.TestCase):
    def setUp(self):
        self.scene = MagicMock()
        self.scene.add_entity.return_value = "mock_entity_handle"

        self.generator = TaskGenerator()

    def test_reset_task_creates_object_and_instruction(self):
        # Run reset_task
        self.generator.reset_task(self.scene)

        # Verify an entity was added
        self.scene.add_entity.assert_called_once()
        args, _ = self.scene.add_entity.call_args
        morph = args[0]
        self.assertIn(morph, ["mock_box_morph", "mock_sphere_morph", "mock_cylinder_morph"])

        # Verify instruction is set
        instruction, entity = self.generator.get_instruction()
        self.assertIsNotNone(instruction)
        self.assertIsInstance(instruction, str)
        self.assertGreater(len(instruction), 0)

        # Verify entity is correct
        self.assertEqual(entity, "mock_entity_handle")

        # Verify instruction contains relevant words
        # Since randomization is involved, we can't be 100% sure which object was picked,
        # but we can check if it contains a color or object name from DB
        found_object = False
        for obj_name in self.generator.ASSET_DB.keys():
            if obj_name in instruction:
                found_object = True
                break

        # Also check for "item" which is used in one of the templates
        if "item" in instruction:
            found_object = True

        self.assertTrue(found_object, f"Instruction '{instruction}' does not contain any known object name")

    def test_reset_task_position_bounds(self):
        # We need to capture the arguments passed to the morph constructor to verify position
        # Since we use lambdas in ASSET_DB, the morph constructor is called inside reset_task

        # Let's patch the morph constructors to inspect arguments
        with patch("genesis.morphs.Box") as mock_box, \
             patch("genesis.morphs.Sphere") as mock_sphere, \
             patch("genesis.morphs.Cylinder") as mock_cylinder:

            # Configure side effects to return dummy morphs so add_entity doesn't fail
            mock_box.return_value = "mock_box_morph"
            mock_sphere.return_value = "mock_sphere_morph"
            mock_cylinder.return_value = "mock_cylinder_morph"

            self.generator.reset_task(self.scene)

            # Find which mock was called
            called_mock = None
            if mock_box.called:
                called_mock = mock_box
            elif mock_sphere.called:
                called_mock = mock_sphere
            elif mock_cylinder.called:
                called_mock = mock_cylinder

            self.assertIsNotNone(called_mock, "No morph constructor was called")

            # Get arguments
            args, kwargs = called_mock.call_args
            # We used keyword arguments in lambda: Box(pos=pos, ...)
            pos = kwargs.get('pos')

            self.assertIsNotNone(pos, "Position argument missing")
            x, y, z = pos

            # Verify bounds
            self.assertTrue(0.3 <= x <= 0.7, f"x={x} out of bounds [0.3, 0.7]")
            self.assertTrue(-0.2 <= y <= 0.2, f"y={y} out of bounds [-0.2, 0.2]")
            self.assertEqual(z, 0.05, f"z={z} expected 0.05")

    def test_clear_previous_object(self):
        # First call to set an object
        self.generator.reset_task(self.scene)
        first_entity = self.generator.target_object_entity
        self.assertIsNotNone(first_entity)

        # Configure remove_entity mock
        self.scene.remove_entity = MagicMock()

        # Second call should trigger removal
        self.generator.reset_task(self.scene)

        # Verify remove_entity called with first_entity
        self.scene.remove_entity.assert_called_with(first_entity)

        # Verify new entity is different (or at least stored)
        self.assertIsNotNone(self.generator.target_object_entity)

if __name__ == "__main__":
    unittest.main()
