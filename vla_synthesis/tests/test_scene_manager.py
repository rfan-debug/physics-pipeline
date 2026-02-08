import sys
from unittest.mock import MagicMock

# Mock genesis before importing SceneManager
mock_gs = MagicMock()
sys.modules['genesis'] = mock_gs

import unittest
import numpy as np

# Adjust path to allow importing from src
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from vla_synthesis.src.scene_manager import SceneManager

class TestSceneManager(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_gs.reset_mock()

        # Setup Scene mock
        self.mock_scene = MagicMock()
        mock_gs.Scene.return_value = self.mock_scene

        # Setup Camera mock
        self.mock_camera = MagicMock()
        self.mock_scene.add_camera.return_value = self.mock_camera

        # Setup Light mock (return value of add_entity when light is added)
        self.mock_light = MagicMock()

        # We need to distinguish between robot and light calls to add_entity if possible
        # Or just let them return a generic mock
        self.mock_entity = MagicMock()
        self.mock_scene.add_entity.return_value = self.mock_entity

    def test_init(self):
        manager = SceneManager(debug=True)
        mock_gs.init.assert_called_with(backend=mock_gs.gpu)
        mock_gs.Scene.assert_called_with(show_viewer=True)
        self.assertEqual(manager.scene, self.mock_scene)

    def test_load_robot(self):
        manager = SceneManager()
        manager.load_robot()

        # Check add_entity called for plane and robot
        # We expect at least 2 calls
        self.assertGreaterEqual(self.mock_scene.add_entity.call_count, 2)

        # Check that one call used MJCF
        # arguments to add_entity are (entity_object,)
        # so we check if any call args[0] was an MJCF object
        # which is created via gs.morphs.MJCF
        mock_gs.morphs.MJCF.assert_called()

    def test_setup_camera(self):
        manager = SceneManager()
        manager.setup_camera()

        self.mock_scene.add_camera.assert_called_once()
        _, kwargs = self.mock_scene.add_camera.call_args

        self.assertIn('res', kwargs)
        self.assertIn('pos', kwargs)
        self.assertIn('lookat', kwargs)

        # Check randomization
        # We can't check exact values easily, but we can check they are within range
        # base_pos = [1.0, 0.0, 0.8]
        # range +/- 0.05
        pos = kwargs['pos']
        self.assertTrue(0.95 <= pos[0] <= 1.05)
        self.assertTrue(-0.05 <= pos[1] <= 0.05)
        self.assertTrue(0.75 <= pos[2] <= 0.85)

        # lookat target = [0.5, 0.0, 0.0]
        # range +/- 0.02
        lookat = kwargs['lookat']
        self.assertTrue(0.48 <= lookat[0] <= 0.52)
        self.assertTrue(-0.02 <= lookat[1] <= 0.02)
        self.assertTrue(-0.02 <= lookat[2] <= 0.02)

    def test_randomize_lighting(self):
        manager = SceneManager()
        manager.randomize_lighting()

        # Check light creation
        mock_gs.morphs.Light.assert_called()
        self.mock_scene.add_entity.assert_called()

    def test_step(self):
        manager = SceneManager()
        manager.step()
        self.mock_scene.step.assert_called_once()

    def test_render(self):
        manager = SceneManager()

        # Mock camera methods
        self.mock_camera.get_color.return_value = np.zeros((480, 640, 3))
        self.mock_camera.get_depth.return_value = np.zeros((480, 640))
        self.mock_camera.get_segmentation.return_value = np.zeros((480, 640))

        # Must setup camera first
        manager.setup_camera()

        rgb, depth, seg = manager.render()

        self.mock_camera.render.assert_called_once()
        self.assertIsNotNone(rgb)
        self.assertIsNotNone(depth)
        self.assertIsNotNone(seg)

    def test_render_no_camera(self):
        manager = SceneManager()
        with self.assertRaises(RuntimeError):
            manager.render()

if __name__ == '__main__':
    unittest.main()
